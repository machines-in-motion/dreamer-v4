import sys
import argparse
import time
import torch
import numpy as np
import cv2
import threading
import math
from abc import ABC, abstractmethod
#from pynput import keyboard
import struct
import socket
from dataclasses import dataclass
import subprocess
import shutil
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image

# --- Models ---
from models import DreamerV4Encoder, DreamerV4Decoder, DreamerV4Dynamics

# --- Configuration ---
IMG_H, IMG_W = 128, 128
PATCH = 16
N_LATENTS = 256
BOTTLENECK_D = 16
D_MODEL_ENC = 768
N_LAYERS_ENC = 12
HEADS_Q_ENC = 12
HEADS_KV_LATENT_ENC = 12
MLP_RATIO = 4.0
TEMPORAL_EVERY = 4
D_MODEL_DYN = 2048 # 1536
N_LAYERS_DYN = 32 # 24
HEADS_Q_DYN = 32 # 24
NUM_REGISTERS = 4
NUM_TAU_LEVELS = 128
CONTEXT_T_DYN = 32
MAX_INTERACTIVE_LEN = 5000
SEQ_COR_TAU_IDX = 12
DINO_MODEL = "dinov2_vits14"
TRAIN_REWARD_MODEL = True
MTP_LENGTH = 8

def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    elif torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")

# ---------------------------------------------------------
# REWARD MODEL DEFINITION
# ---------------------------------------------------------
class ValueTransformerWithInitial(nn.Module):
    def __init__(self, embedding_dim=384, num_patches=256, num_layers=4, num_heads=8, dropout=0.05):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.token_count = 3 * num_patches + 2 
        self.start_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.end_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.token_count, embedding_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.value_head = nn.Sequential(nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, 1))

    def forward(self, dino_current, dino_start, dino_goal):
        B = dino_current.shape[0]
        start_token = self.start_token.expand(B, -1, -1)
        end_token = self.end_token.expand(B, -1, -1)
        x = torch.cat([start_token, dino_current, dino_start, dino_goal, end_token], dim=1)
        x = x + self.pos_embedding[:, :x.shape[1], :]
        x = self.transformer(x)
        return self.value_head(x[:, -1])

class GstServer:
    def __init__(self, port=5000, width=128, height=128, fps=24):
        if not shutil.which("gst-launch-1.0"):
            print("Error: gst-launch-1.0 not found. Install GStreamer.")
            self.proc = None
            return

        # Pipeline: Raw BGR -> x264 (Low Latency) -> TCP Server
        self.cmd = (
            f"gst-launch-1.0 -q fdsrc ! "
            f"videoparse format=bgr width={width} height={height} framerate={int(fps)}/1 ! "
            "videoconvert ! "
            "x264enc tune=zerolatency speed-preset=ultrafast key-int-max=30 bitrate=500 ! "
            "video/x-h264,stream-format=byte-stream ! "
            "h264parse config-interval=1 ! "
            "queue leaky=downstream max-size-buffers=1 ! "
            f"tcpserversink host=0.0.0.0 port={port} sync-method=latest-keyframe"
        )
        
        print(f"Starting GStreamer TCP Server on port {port} (FPS: {fps})...")
        self.proc = subprocess.Popen(self.cmd.split(), stdin=subprocess.PIPE)

    def write(self, frame):
        """Frame must be a (H, W, 3) uint8 numpy array (BGR)"""
        if self.proc and self.proc.stdin:
            try:
                self.proc.stdin.write(frame.tobytes())
                self.proc.stdin.flush()
            except BrokenPipeError:
                print("GStreamer pipe broken.")
                self.close()

    def close(self):
        if self.proc:
            if self.proc.stdin: self.proc.stdin.close()
            self.proc.terminate()
            self.proc.wait()
            self.proc = None


# ---------------------------------------------------------
# 1. MOSAIC SELECTOR
# ---------------------------------------------------------
class TrajectorySelector:
    def __init__(self, images_tensor, context_len):
        self.images = images_tensor
        self.B = images_tensor.shape[0]
        self.selected_index = None
        self.context_len = context_len
        
        # Mosaic Layout
        self.cols = 8
        self.rows = math.ceil(self.B / self.cols)
        self.thumb_h, self.thumb_w = IMG_H, IMG_W
        
        self.canvas = np.zeros((self.rows * self.thumb_h, self.cols * self.thumb_w, 3), dtype=np.uint8)
        
        for idx in range(self.B):
            img = self.images[idx, self.context_len-1].float().permute(1, 2, 0).numpy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            
            r, c = divmod(idx, self.cols)
            y_off = r * self.thumb_h
            x_off = c * self.thumb_w
            
            cv2.putText(img, str(idx), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            self.canvas[y_off:y_off+self.thumb_h, x_off:x_off+self.thumb_w] = img

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            col_idx = x // self.thumb_w
            row_idx = y // self.thumb_h
            index = row_idx * self.cols + col_idx
            
            if index < self.B:
                self.selected_index = index
                print(f"Selected Trajectory: {index}")

    def select(self):
        window_name = "Select Trajectory (Click one)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse_callback)
        
        print("Displaying mosaic. Click on an image to select it.")
        
        while self.selected_index is None:
            cv2.imshow(window_name, self.canvas)
            key = cv2.waitKey(50)
            if key == 27 or key == ord('q'): 
                print("Selection cancelled.")
                sys.exit(0)
                
        cv2.destroyWindow(window_name)
        return self.selected_index

# ---------------------------------------------------------
# 2. CONTROLLERS
# ---------------------------------------------------------
class Controller(ABC):
    def __init__(self, action_dim=3): # Changed default to 3 for Go2
        self.action_dim = action_dim
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        # Go2 Actions: [x_vel, y_vel, yaw_vel]
        self._current_action = np.zeros(action_dim, dtype=np.float32)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
        print(f"{self.__class__.__name__} stopped.")

    @abstractmethod
    def _run_loop(self): pass

    def get_action(self):
        with self.lock:
            # Return current state (Zero-Order Hold)
            return self._current_action.copy()

class KeyboardController(Controller):
    def __init__(self, action_dim=3, sensitivity=0.5):
        super().__init__(action_dim)
        self.sensitivity = sensitivity
        self.listener = None 
        # Key Map for Go2
        # Action 0: X (Forward/Back) -> Up/Down Arrows
        # Action 1: Y (Strafe Left/Right) -> Left/Right Arrows
        # Action 2: Yaw (Turn Left/Right) -> A/D Keys
        self.pressed_keys = set()

    def _on_press(self, key):
        try: k = key.char 
        except AttributeError: k = key 
        self.pressed_keys.add(k)

    def _on_release(self, key):
        try: k = key.char 
        except AttributeError: k = key 
        if k in self.pressed_keys:
            self.pressed_keys.remove(k)

    def _run_loop(self):
        # We start the listener in non-blocking mode
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()
        
        while self.running:
            # Poll keys to update velocity vector
            vx, vy, vyaw = 0.0, 0.0, 0.0
            
            if keyboard.Key.up in self.pressed_keys: vx += 1.0
            if keyboard.Key.down in self.pressed_keys: vx -= 1.0
            
            if keyboard.Key.left in self.pressed_keys: vy += 1.0 # Go2 convention: +Y is Left
            if keyboard.Key.right in self.pressed_keys: vy -= 1.0
            
            if 'a' in self.pressed_keys: vyaw += 1.0
            if 'd' in self.pressed_keys: vyaw -= 1.0
            
            with self.lock:
                self._current_action[0] = vx * self.sensitivity
                self._current_action[1] = vy * self.sensitivity
                self._current_action[2] = vyaw * self.sensitivity
            
            time.sleep(0.01)

    def stop(self):
        if self.listener is not None:
            self.listener.stop()
        super().stop()

@dataclass
class ControllerPose:
    matrix: np.ndarray 

@dataclass
class ControllerInput:
    joystick: tuple[float, float]
    index_trigger: float
    hand_trigger: float
    buttons: dict[str, bool]

@dataclass
class VRFrame:
    head_pose: ControllerPose
    left_pose: ControllerPose
    right_pose: ControllerPose
    left_input: ControllerInput
    right_input: ControllerInput

class VRController(Controller):
    def __init__(self, action_dim=3, ip="0.0.0.0", port=5000, max_delta=1.0):
        super().__init__(action_dim)
        self.ip = ip
        self.port = port
        self.max_delta = max_delta # Scaling factor for joysticks
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.ip, self.port))
        self.sock.settimeout(1.0) 
        
        # State tracking for rising edge detection
        self.last_right_grip = False
        self.last_right_index = False
        
        # Flags for the Generator to read
        self.set_start_flag = False
        self.set_goal_flag = False

        self.remote_addr = None
        print(f"VRController listening on {self.ip}:{self.port}")

    def _decode_packet(self, data: bytes) -> VRFrame:
        if len(data) != 66 * 4: return None
        floats = struct.unpack('<66f', data)
        f = iter(floats)

        def next_mat():
            return np.array([ 
                [next(f), next(f), next(f), next(f)],
                [next(f), next(f), next(f), next(f)],
                [next(f), next(f), next(f), next(f)],
                [next(f), next(f), next(f), next(f)] 
            ], dtype=np.float32)

        head_pose  = ControllerPose(next_mat())
        left_pose  = ControllerPose(next_mat())
        right_pose = ControllerPose(next_mat())

        left_joy   = (next(f), next(f))
        right_joy  = (next(f), next(f))
        left_index, right_index, left_grip, right_grip = next(f), next(f), next(f), next(f)
        
        # Buttons
        btnA, btnB, btnX, btnY, thumbL, thumbR, trigL, trigR, gripL, gripR = [int(next(f)) for _ in range(10)]
        
        left_input  = ControllerInput(left_joy, left_index, left_grip, {})
        right_input = ControllerInput(right_joy, right_index, right_grip, {})

        return VRFrame(head_pose, left_pose, right_pose, left_input, right_input)

    def _run_loop(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(4096)
                self.remote_addr = addr 
                frame = self._decode_packet(data)
                if frame is None: continue
                
                # --- Go2 Teleop Logic ---
                left_x, left_y = frame.left_input.joystick
                right_x, right_y = frame.right_input.joystick
                
                # Apply Deadzone
                deadzone = 0.1
                if abs(left_x) < deadzone: left_x = 0
                if abs(left_y) < deadzone: left_y = 0
                if abs(right_x) < deadzone: right_x = 0
                
                val_x = left_x
                val_y = left_y
                val_yaw = right_x
                
                # --- DETECT RISING EDGES ---
                curr_grip = frame.right_input.hand_trigger > 0.5
                curr_index = frame.right_input.index_trigger > 0.5

                with self.lock:
                    # Set flag True if pressed now AND wasn't pressed before
                    if curr_grip and not self.last_right_grip:
                        self.set_start_flag = True

                    if curr_index and not self.last_right_index:
                        self.set_goal_flag = True

                    self._current_action[0] = val_x * self.max_delta
                    self._current_action[1] = val_y * self.max_delta
                    self._current_action[2] = val_yaw * self.max_delta

                self.last_right_grip = curr_grip
                self.last_right_index = curr_index

            except socket.timeout:
                continue
            except Exception as e:
                print(f"VR Controller Error: {e}")

    def pop_flags(self):
        with self.lock:
            s, g = self.set_start_flag, self.set_goal_flag
            self.set_start_flag = False
            self.set_goal_flag = False
            return s, g

    def send_image(self, img_np):
        if self.remote_addr is None: return
        success, buffer = cv2.imencode(".jpg", img_np, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        if success:
            try:
                self.sock.sendto(buffer.tobytes(), self.remote_addr)
            except Exception: pass

    def stop(self):
        super().stop()
        self.sock.close()

# ---------------------------------------------------------
# 3. TRAJECTORY GENERATOR
# ---------------------------------------------------------
class TrajectoryGenerator:
    def __init__(self, args, device, selected_images, selected_actions, fps=5.0):
        self.args = args
        self.device = device
        self.running = False
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        self.v0 = None  # Value of the start frame
        self.vN = None  # Value of the goal frame

        # --- FPS Control ---
        self.target_fps = fps
        self.frame_duration = 1.0 / self.target_fps
        print(f"Generator configured for {self.target_fps} Hz")
        # -------------------
        
        # Data
        self.images_ctx = selected_images
        self.actions_ctx = selected_actions
        self.B = 1
        self.action_dim = self.actions_ctx.shape[-1]
        
        self._init_models()
        self._init_context()

        # Reward Model Setup
        self.use_reward = args.use_reward
        if self.use_reward:
            self._init_reward_model(args.reward_ckpt)
            
            # Storage for start/goal DINO features
            self.s0_feat = None
            self.sg_feat = None
            
            # Normalization for DINO
            self.dino_norm = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            # Load Target Image from Disk if provided
            if args.target_image:
                print(f"Loading target image from {args.target_image}...")
                try:
                    pil_img = Image.open(args.target_image).convert("RGB")
                    tensor_img = T.ToTensor()(pil_img).unsqueeze(0).to(self.device).to(torch.bfloat16)
                    
                    # Compute Feature
                    self.sg_feat = self._get_dino_features(tensor_img)
                    print("Goal state (sg) initialized from file.")
                except Exception as e:
                    print(f"Error loading target image: {e}")

        # Initialize streaming with correct FPS for metadata
        self.gst_server = GstServer(port=5000, width=IMG_W, height=IMG_H, fps=self.target_fps)

    def _init_reward_model(self, ckpt_path):
        print("Loading Reward Model & DINO...")
        # 1. Load DINO
        self.dino = torch.hub.load("facebookresearch/dinov2", DINO_MODEL).to(self.device)
        self.dino = self.dino.to(dtype=torch.bfloat16)
        self.dino.eval()
        
        # 2. Load Reward Network
        self.reward_model = ValueTransformerWithInitial().to(self.device)
        self.reward_model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.reward_model = self.reward_model.to(dtype=torch.bfloat16)
        self.reward_model.eval()
        # Optional: Compile
        # self.reward_model = torch.compile(self.reward_model)

    # Helper to get DINO features from a frame
    @torch.no_grad()
    def _get_dino_features(self, img_tensor):
        # img_tensor: [1, 3, H, W] (Float 0-1)
        # Resize to 224x224 for DINO
        imgs_resized = torch.nn.functional.interpolate(
            img_tensor, size=(224, 224), mode="bilinear", align_corners=False
        )
        imgs_norm = self.dino_norm(imgs_resized)
        out = self.dino.forward_features(imgs_norm)
        return out['x_norm_patchtokens'] # [1, 256, 384]

    def _init_models(self):
        print("Loading models...")
        ckpt_tok = torch.load(self.args.tokenizer_ckpt, map_location="cpu")
        ckpt_dyn = torch.load(self.args.dynamics_ckpt, map_location="cpu")

        self.enc = DreamerV4Encoder(
            image_size=(IMG_H, IMG_W), patch_size=PATCH, d_model=D_MODEL_ENC,
            n_layers=N_LAYERS_ENC, num_heads_q=HEADS_Q_ENC, num_heads_kv_latent=HEADS_KV_LATENT_ENC,
            seq_len=MAX_INTERACTIVE_LEN, mlp_ratio=MLP_RATIO, n_latents=N_LATENTS,
            bottleneck_dim=BOTTLENECK_D, temporal_every=TEMPORAL_EVERY
        ).to(self.device, dtype=torch.bfloat16).eval()
        self.enc = torch.compile(self.enc)
        self.enc.load_state_dict(ckpt_tok["enc"])

        self.dec = DreamerV4Decoder(
            image_size=(IMG_H, IMG_W), patch_size=PATCH, d_model=D_MODEL_ENC,
            n_layers=N_LAYERS_ENC, num_heads_q=HEADS_Q_ENC, num_heads_kv_latent=HEADS_KV_LATENT_ENC,
            bottleneck_dim=BOTTLENECK_D, seq_len=MAX_INTERACTIVE_LEN, mlp_ratio=MLP_RATIO,
            n_latents=N_LATENTS, temporal_every=TEMPORAL_EVERY
        ).to(self.device, dtype=torch.bfloat16).eval()
        self.dec = torch.compile(self.dec)
        if "dec" in ckpt_tok: self.dec.load_state_dict(ckpt_tok["dec"])
        
        self.dyn = DreamerV4Dynamics(
            action_dim=self.action_dim, num_latents=N_LATENTS, latent_dim=BOTTLENECK_D,
            d_model=D_MODEL_DYN, num_layers=N_LAYERS_DYN, num_heads=HEADS_Q_DYN,
            num_registers=NUM_REGISTERS, seq_len=MAX_INTERACTIVE_LEN, num_tau_levels=NUM_TAU_LEVELS,
            temporal_every=TEMPORAL_EVERY, train_reward_model=TRAIN_REWARD_MODEL, mtp_length=MTP_LENGTH
        ).to(self.device, dtype=torch.bfloat16).eval()
        self.dyn = torch.compile(self.dyn)
        self.dyn.load_state_dict(ckpt_dyn["dyn"])

    def _init_context(self):
        self.dyn.init_cache(self.B, self.device, max_seq_len=CONTEXT_T_DYN)
        self.dec.init_cache(self.B, self.device, max_seq_len=CONTEXT_T_DYN)
        d_min_idx = int(np.log2(NUM_TAU_LEVELS))
        
        print(f"Prefilling context (0 to {self.args.context_len})...")
        with torch.no_grad():
            _, _, z_gt = self.enc(self.images_ctx)
            context_z = z_gt[:, :self.args.context_len]
            context_actions = self.actions_ctx[:, :self.args.context_len]

            dummy_sigma = torch.zeros((self.B, self.args.context_len), dtype=torch.long, device=self.device)
            dummy_step = torch.full((self.B, self.args.context_len), d_min_idx, dtype=torch.long, device=self.device)

            self.dyn.forward_step(
                action=context_actions, noisy_z=context_z, sigma_idx=dummy_sigma,
                step_idx=dummy_step, start_step_idx=0, update_cache=True
            )
            self.dec.forward_step(context_z, start_step_idx=0, update_cache=True)
            self.current_z = context_z[:, -1:]
            self.t = self.args.context_len

    @torch.no_grad()
    def _solve_frame(self, actions_t, z_gen, t, num_steps=4):
        B, _, N, D = z_gen.shape
        z_t = torch.randn(B, 1, N, D, device=self.device, dtype=torch.bfloat16)
        step_val = 1 / num_steps
        step_idx = int(np.log2(num_steps))
        d_min_idx = int(np.log2(NUM_TAU_LEVELS))
        
        for i in range(num_steps):
            tau_curr = i / num_steps
            curr_tau_idx = int(((num_steps - 1 - i) + 1)*(2**(d_min_idx-step_idx)))-1
            tau_idxs = torch.full((B, 1), curr_tau_idx, dtype=torch.long, device=self.device)
            step_idxs = torch.full((B, 1), step_idx, dtype=torch.long, device=self.device)

            if TRAIN_REWARD_MODEL:
                pred, final_reward_logits = self.dyn.forward_step(
                    action=actions_t, noisy_z=z_t, sigma_idx=tau_idxs,
                    step_idx=step_idxs, start_step_idx=t, update_cache=False
                )
            else:
                pred = self.dyn.forward_step(
                    action=actions_t, noisy_z=z_t, sigma_idx=tau_idxs,
                    step_idx=step_idxs, start_step_idx=t, update_cache=False
                )
            z_t = z_t + (pred - z_t) / max(1.0 - tau_curr, 1e-5) * step_val

        tau_idxs = torch.full((B, 1), SEQ_COR_TAU_IDX, dtype=torch.long, device=self.device)
        step_idxs = torch.full((B, 1), d_min_idx, dtype=torch.long, device=self.device)
        seq_cor_tau = torch.full((B, 1, 1, 1), 1. - ((SEQ_COR_TAU_IDX + 1) / NUM_TAU_LEVELS), dtype=torch.bfloat16, device=self.device)
        eps = torch.randn_like(z_t)
        cor_z_t = (1. - seq_cor_tau) * eps + seq_cor_tau * z_t
        self.dyn.forward_step(
            action=actions_t, noisy_z=cor_z_t, sigma_idx=tau_idxs,
            step_idx=step_idxs, start_step_idx=t, update_cache=True
        )

        # --- Convert Reward Logits to Scalar ---
        scalar_reward = 0.0
        if TRAIN_REWARD_MODEL:
            # 1. Access the buckets buffer from the model
            #    The RewardMTPHead has a list of heads; we use the first one to get the buckets
            #    Structure: dyn.reward_head.heads[0].buckets
            #    buckets shape: (NUM_BUCKETS,) e.g. [-20, ..., 20]
            buckets = self.dyn.reward_head.heads[0].buckets
            
            # 2. Compute probabilities via Softmax
            #    logits: (B, 1, MTP_LENGTH, NUM_BUCKETS) -> probs: (B, 1, MTP_LENGTH, NUM_BUCKETS)
            probs = torch.softmax(final_reward_logits, dim=-1)
            
            # 3. Expected Value in Symlog Space
            #    Sum(prob * bucket_value) over the last dimension
            #    buckets needs to be broadcastable: (1, 1, 1, NUM_BUCKETS)
            buckets_view = buckets.view(1, 1, 1, -1).to(probs.device).to(probs.dtype)
            expected_symlog = (probs * buckets_view).sum(dim=-1) # (B, 1, MTP_LENGTH)
            
            # 4. Convert Symlog -> Real Value
            #    We use the helper function from the first head
            expected_value = self.dyn.reward_head.heads[0].from_symlog(expected_symlog)
            
            # 5. Select the immediate reward (step 0 of MTP)
            #    Shape is (B, 1, MTP_LENGTH). We take index 0 for the immediate next step.
            #    You could also average them if you wanted a "value" estimate over the horizon.
            scalar_reward = expected_value[0, 0, 0].item()

        if TRAIN_REWARD_MODEL:
            return z_t, scalar_reward
        else:
            return z_t

    def start(self, controller):
        self.running = True
        self.thread = threading.Thread(target=self._loop, args=(controller,), daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread: self.thread.join()
        if hasattr(self, 'gst_server'):
            self.gst_server.close()

    def _loop(self, controller):
        print("Generator loop started...")
        while self.running and self.t < MAX_INTERACTIVE_LEN:
            loop_start = time.time()
            action_np = controller.get_action()
            action_t = torch.from_numpy(action_np).unsqueeze(0).unsqueeze(0).to(self.device).to(torch.bfloat16)

            reward_score = 0.0
            norm_score = 0.0 # Variable for normalized score

            if TRAIN_REWARD_MODEL:
                z_next, norm_score = self._solve_frame(action_t, self.current_z, self.t)
            else:
                z_next = self._solve_frame(action_t, self.current_z, self.t)
            with torch.no_grad():
                _, recon_frame = self.dec.forward_step(z_next, start_step_idx=self.t, update_cache=True)

            # --- REWARD LOGIC START ---
            if self.use_reward and hasattr(controller, 'pop_flags'):
                set_s0, set_sg = controller.pop_flags()
                
                # FIX: Ensure 4D Tensor [B, C, H, W]
                curr_img_t = recon_frame[:, 0]
                curr_feat = self._get_dino_features(curr_img_t)
                
                # 3. Update Anchor States
                if set_s0:
                    self.s0_feat = curr_feat.clone()
                    self.v0 = None # Reset normalization baseline
                    print("Updated Start State (s0)")
                    
                if set_sg:
                    self.sg_feat = curr_feat.clone()
                    self.vN = None # Reset normalization max
                    print("Updated Goal State (sg)")
                
                # 4. Compute Reward & Normalization
                if self.s0_feat is not None and self.sg_feat is not None:
                    # A. Compute V(current)
                    v_curr = self.reward_model(curr_feat, self.s0_feat, self.sg_feat).item()
                    reward_score = v_curr
                    
                    # B. Compute/Cache Baselines (v0 and vN)
                    # We compute these ONCE when anchors change to avoid redundant compute
                    if self.v0 is None:
                        # V(s0) should theoretically be low (distance) or high (similarity)?
                        # Assuming your model outputs negative distance or similarity.
                        # We pass s0 as "current", s0 as "start", sg as "goal"
                        self.v0 = self.reward_model(self.s0_feat, self.s0_feat, self.sg_feat).item()
                        
                    if self.vN is None:
                        # V(sg) 
                        self.vN = self.reward_model(self.sg_feat, self.s0_feat, self.sg_feat).item()
                    
                    # C. Normalize
                    # Avoid division by zero
                    denominator = self.vN - self.v0
                    if abs(denominator) > 1e-6:
                        norm_score = (v_curr - self.v0) / denominator
                    else:
                        norm_score = 0.0

            gen_frame = recon_frame[0, 0].float().cpu().permute(1, 2, 0).numpy()
            display_img = cv2.cvtColor(gen_frame, cv2.COLOR_RGB2BGR)
            display_img = np.clip(display_img * 255, 0, 255).astype(np.uint8)

            # --- VISUALIZATION OVERLAY ---
            if self.use_reward or TRAIN_REWARD_MODEL:
                # Display NORMALIZED Score
                # Green if > 0.5 (closer to goal), Red if < 0.5
                color = (0, 255, 0) if norm_score > 0.5 else (0, 0, 255)
                
                # Show Raw and Normalized
                text = f"N: {norm_score:.2f}"
                cv2.putText(display_img, text, (5, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                # Status Indicators
                if not TRAIN_REWARD_MODEL and self.s0_feat is not None:
                    cv2.circle(display_img, (10, 10), 4, (255, 0, 0), -1) # Blue dot = s0 set
                if not TRAIN_REWARD_MODEL and self.sg_feat is not None:
                    cv2.circle(display_img, (20, 10), 4, (0, 255, 255), -1) # Yellow dot = sg set
            # --- REWARD LOGIC END ---

            with self.frame_lock: self.latest_frame = display_img

            if hasattr(controller, 'send_image'):
                controller.send_image(display_img)
            if hasattr(self, 'gst_server'):
                self.gst_server.write(display_img)

            self.current_z = z_next
            self.t += 1
            
            elapsed = time.time() - loop_start
            time.sleep(max(0.0, self.frame_duration - elapsed))

# ---------------------------------------------------------
# 4. MAIN
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_ckpt", type=str, required=True)
    parser.add_argument("--dynamics_ckpt", type=str, required=True)
    parser.add_argument("--traj_path", type=str, required=True)
    parser.add_argument("--context_len", type=int, default=32)
    parser.add_argument("--use_vr", action="store_true", help="Use VR Controller")
    parser.add_argument("--fps", type=float, default=5.0, help="Simulation FPS (default: 5.0)")
    parser.add_argument("--use_reward", action="store_true", help="Enable reward model visualization")
    parser.add_argument("--reward_ckpt", type=str, default="reward_model.pt", help="Path to reward model checkpoint")
    parser.add_argument("--target_image", type=str, default=None, help="Path to initial goal image (png/jpg)")

    args = parser.parse_args()
    device = get_device()
    
    print(f"Loading trajectories from {args.traj_path}...")
    traj_data = torch.load(args.traj_path, map_location="cpu")
    
    if isinstance(traj_data, dict):
        all_images = traj_data['images'] if 'images' in traj_data else traj_data['image']
        all_actions = traj_data['actions'] if 'actions' in traj_data else traj_data['action']
    else:
        print("Error: trajectory file format unknown.")
        return

    selector = TrajectorySelector(all_images, args.context_len)
    idx = selector.select()
    
    print(f"Initializing generator with trajectory {idx}...")
    selected_images = all_images[idx:idx+1].to(device).to(torch.bfloat16)
    selected_actions = all_actions[idx:idx+1].to(device).to(torch.bfloat16)
    
    # Pass custom FPS here
    generator = TrajectoryGenerator(args, device, selected_images, selected_actions, fps=args.fps)
    
    if args.use_vr:
         controller = VRController(action_dim=generator.action_dim)
    else:
         controller = KeyboardController(action_dim=generator.action_dim)

    controller.start()
    generator.start(controller)
    
    window_name = "Interactive Generation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    try:
        while True:
            with generator.frame_lock:
                frame = generator.latest_frame
            
            if frame is not None:
                cv2.imshow(window_name, frame)
            
            key = cv2.waitKey(10)
            if key == ord('q') or key == 27:
                break
            
            if not generator.running:
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping...")
        controller.stop()
        generator.stop()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        print("Done.")

if __name__ == "__main__":
    main()
