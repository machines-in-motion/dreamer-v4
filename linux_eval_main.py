import sys
import argparse
import time
import torch
import numpy as np
import cv2
import threading
import math
from abc import ABC, abstractmethod
from pynput import keyboard
import struct
import socket
from dataclasses import dataclass

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
D_MODEL_DYN = 1536
N_LAYERS_DYN = 24
HEADS_Q_DYN = 24
NUM_REGISTERS = 4
NUM_TAU_LEVELS = 128
CONTEXT_T_DYN = 32
MAX_INTERACTIVE_LEN = 5000
SEQ_COR_TAU_IDX = 12

def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    elif torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")

# ---------------------------------------------------------
# 1. MOSAIC SELECTOR
# ---------------------------------------------------------
class TrajectorySelector:
    def __init__(self, images_tensor):
        """
        images_tensor: [B, T, C, H, W] (CPU)
        """
        self.images = images_tensor
        self.B = images_tensor.shape[0]
        self.selected_index = None
        
        # Mosaic Layout
        self.cols = 8
        self.rows = math.ceil(self.B / self.cols)
        self.thumb_h, self.thumb_w = IMG_H, IMG_W
        
        # Create the big canvas
        self.canvas = np.zeros((self.rows * self.thumb_h, self.cols * self.thumb_w, 3), dtype=np.uint8)
        
        # Populate canvas
        for idx in range(self.B):
            # Get first frame: [C, H, W] -> [H, W, C] -> BGR
            img = self.images[idx, 0].float().permute(1, 2, 0).numpy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            
            r, c = divmod(idx, self.cols)
            y_off = r * self.thumb_h
            x_off = c * self.thumb_w
            
            # Add index text
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
            if key == 27 or key == ord('q'): # Esc/Quit
                print("Selection cancelled.")
                sys.exit(0)
                
        cv2.destroyWindow(window_name)
        return self.selected_index

# ---------------------------------------------------------
# 2. CONTROLLERS
# ---------------------------------------------------------
class Controller(ABC):
    def __init__(self, action_dim):
        self.action_dim = action_dim
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self._current_action = np.zeros(action_dim, dtype=np.float32)
        self._gripper_state = -1.0 
        self._current_action[6] = self._gripper_state

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Base stop stops the wrapper thread."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
        print(f"{self.__class__.__name__} stopped.")

    @abstractmethod
    def _run_loop(self): pass

    def get_action(self):
        with self.lock:
            action_to_return = self._current_action.copy()
            self._current_action[:6] = 0.0
            self._current_action[6] = self._gripper_state
            return action_to_return

class KeyboardController(Controller):
    def __init__(self, action_dim=7, sensitivity=0.01):
        super().__init__(action_dim)
        self.sensitivity = sensitivity
        self.listener = None # Store reference to listener
        self.key_map = {
            keyboard.Key.up: (0, 1), keyboard.Key.down: (0, -1),
            keyboard.Key.left: (1, 1), keyboard.Key.right: (1, -1),
            'w': (2, 1), 's': (2, -1),
        }

    def _on_press(self, key):
        try: k = key.char 
        except AttributeError: k = key 

        with self.lock:
            if k in self.key_map:
                idx, direction = self.key_map[k]
                self._current_action[idx] += direction * self.sensitivity
            elif key == keyboard.Key.space: self._gripper_state = 1.0 
            elif key == keyboard.Key.enter: self._gripper_state = -1.0
            self._current_action[6] = self._gripper_state

    def _run_loop(self):
        # We assign the listener to self.listener so we can stop it externally
        with keyboard.Listener(on_press=self._on_press) as listener:
            self.listener = listener
            listener.join() # This blocks until self.listener.stop() is called

    def stop(self):
        # 1. Stop the internal pynput listener to unblock _run_loop
        if self.listener is not None:
            self.listener.stop()
        
        # 2. Call parent stop to join the thread
        super().stop()

@dataclass
class ControllerPose:
    matrix: np.ndarray  # shape (4,4)

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
    def __init__(self, action_dim=7, ip="0.0.0.0", port=5000, max_delta=0.05):
        super().__init__(action_dim)
        self.ip = ip
        self.port = port
        self.max_delta = max_delta
        
        # UDP Socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.ip, self.port))
        self.sock.settimeout(1.0) # Timeout to allow checking self.running
        
        self.remote_addr = None # Stores (IP, Port) of the Quest

        # State Tracking for Deltas
        self.last_pos = None      # [x, y, z]
        self.last_rot = None      # [roll, pitch, yaw] or quaternion (placeholder for now)
        
        print(f"VRController listening on {self.ip}:{self.port}")

    def _decode_packet(self, data: bytes) -> VRFrame:
        # Expected size: 3 matrices (16 floats) + 8 analog + 10 buttons = 48 + 8 + 10 = 66 floats
        if len(data) != 66 * 4:
            return None
            
        floats = struct.unpack('<66f', data)
        f = iter(floats)

        def next_mat():
            # Unity is usually Column Major, but let's assume standard row fill for now
            # or just treat as 4x4 numpy array.
            return np.array([ 
                [next(f), next(f), next(f), next(f)],
                [next(f), next(f), next(f), next(f)],
                [next(f), next(f), next(f), next(f)],
                [next(f), next(f), next(f), next(f)] 
            ], dtype=np.float32)

        head_pose  = ControllerPose(next_mat())
        left_pose  = ControllerPose(next_mat())
        right_pose = ControllerPose(next_mat())

        # Analog values
        left_joy   = (next(f), next(f))
        right_joy  = (next(f), next(f))
        left_index, right_index, left_grip, right_grip = next(f), next(f), next(f), next(f)

        # Buttons
        btnA, btnB, btnX, btnY, thumbL, thumbR, trigL, trigR, gripL, gripR = [int(next(f)) for _ in range(10)]

        # We construct the frame object (simplified for speed)
        left_buttons = {"TriggerButton": bool(trigL)}
        right_buttons = {"A": bool(btnA)}
        
        left_input  = ControllerInput(left_joy, left_index, left_grip, left_buttons)
        right_input = ControllerInput(right_joy, right_index, right_grip, right_buttons)

        return VRFrame(head_pose, left_pose, right_pose, left_input, right_input)

    def _run_loop(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(4096)
                self.remote_addr = addr 
                frame = self._decode_packet(data)
                if frame is None: continue
                
                # --- Logic to calculate actions ---
                
                # 1. Extract Current Position (Right Controller)
                # Unity Matrix element [0,3], [1,3], [2,3] are usually X, Y, Z translation
                curr_pos = frame.right_pose.matrix[:3, 3] 
                
                # Placeholder for Rotation (if you want to implement rotation deltas later)
                # Extracting Euler angles from 4x4 matrix is slightly complex, 
                # usually involves `scipy.spatial.transform.Rotation`
                curr_rot = np.zeros(3) 

                if frame.right_input.hand_trigger > 0.5:
                    with self.lock:
                        if self.last_pos is not None:
                            # Compute Delta
                            delta_pos = curr_pos - self.last_pos
                            
                            # Unity coordinates might differ from your robot coordinates
                            # e.g., Unity Y is Up, Robot Z might be Up.
                            # Assuming standard mapping for now, swap if necessary:
                            # Robot X = Unity X, Robot Y = Unity Z, Robot Z = Unity Y
                            
                            # Assign to action vector (Indices 0,1,2 are XYZ)
                            # IMPORTANT: Accumulate deltas because get_action resets them,
                            # but we might receive multiple UDP packets between generator steps.
                            self._current_action[0] += delta_pos[2]
                            self._current_action[1] -= delta_pos[0]
                            self._current_action[2] += delta_pos[1]
                            
                            self._current_action = np.clip(self._current_action, -self.max_delta, self.max_delta)
                            # Placeholder for Rotation Deltas (Indices 3,4,5)
                            # delta_rot = curr_rot - self.last_rot
                            # self._current_action[3] += delta_rot[0]
                            # self._current_action[4] += delta_rot[1]
                            # self._current_action[5] += delta_rot[2]
                        
                        # Update Gripper State
                        # Using Right Index Trigger (0.0 to 1.0)
                        # Map 0.0 -> -1 (Closed), 1.0 -> 1 (Open) ?? 
                        # Or usually triggers are 0 (Open) to 1 (Closed).
                        # Let's assume > 0.5 is closed (-1), else open (1)
                        if frame.right_input.index_trigger > 0.5:
                            self._gripper_state = -1.0
                        else:
                            self._gripper_state = 1.0
                        
                        # Sync gripper immediately (it's stateful, not delta)
                        self._current_action[6] = self._gripper_state
                
                # Update history
                self.last_pos = curr_pos
                self.last_rot = curr_rot

            except socket.timeout:
                continue
            except Exception as e:
                print(f"VR Controller Error: {e}")

    def send_image(self, img_np):
        """Compresses and sends the image to the last known Quest address."""
        if self.remote_addr is None: return

        # Compress to JPEG (Quality 80 is a good balance for VR streaming)
        success, buffer = cv2.imencode(".jpg", img_np, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        
        if success:
            try:
                self.sock.sendto(buffer.tobytes(), self.remote_addr)
            except Exception as e:
                print(f"Streaming Error: {e}")

    def stop(self):
        super().stop()
        self.sock.close()

# ---------------------------------------------------------
# 3. TRAJECTORY GENERATOR
# ---------------------------------------------------------
class TrajectoryGenerator:
    def __init__(self, args, device, selected_images, selected_actions):
        """
        selected_images: [1, T, C, H, W] (already sliced and on device)
        selected_actions: [1, T, D]
        """
        self.args = args
        self.device = device
        self.running = False
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.target_fps = 24.0
        self.frame_duration = 1.0 / self.target_fps
        
        # Data
        self.images_ctx = selected_images
        self.actions_ctx = selected_actions
        self.B = 1
        self.action_dim = self.actions_ctx.shape[-1]
        
        self._init_models()
        self._init_context()
        
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
            temporal_every=TEMPORAL_EVERY
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
            is_last_step = (i == num_steps - 1)
            tau_idxs = torch.full((B, 1), curr_tau_idx, dtype=torch.long, device=self.device)
            step_idxs = torch.full((B, 1), step_idx, dtype=torch.long, device=self.device)
            
            pred = self.dyn.forward_step(
                action=actions_t, noisy_z=z_t, sigma_idx=tau_idxs,
                step_idx=step_idxs, start_step_idx=t, update_cache=False
            )
            z_t = z_t + (pred - z_t) / max(1.0 - tau_curr, 1e-5) * step_val

        d_min_idx = int(np.log2(NUM_TAU_LEVELS))
        tau_idxs = torch.full((B, 1), SEQ_COR_TAU_IDX, dtype=torch.long, device=self.device)
        step_idxs = torch.full((B, 1), d_min_idx, dtype=torch.long, device=self.device)
        seq_cor_tau = torch.full((B, 1, 1, 1), 1. - ((SEQ_COR_TAU_IDX + 1) / NUM_TAU_LEVELS), dtype=torch.bfloat16, device=self.device)
        eps = torch.randn_like(z_t)
        cor_z_t = (1. - seq_cor_tau) * eps + seq_cor_tau * z_t
        self.dyn.forward_step(
            action=actions_t, noisy_z=cor_z_t, sigma_idx=tau_idxs,
            step_idx=step_idxs, start_step_idx=t, update_cache=True
        )

        return z_t

    def start(self, controller):
        self.running = True
        self.thread = threading.Thread(target=self._loop, args=(controller,), daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread: self.thread.join()

    def _loop(self, controller):
        print("Generator loop started...")
        while self.running and self.t < MAX_INTERACTIVE_LEN:
            loop_start = time.time()
            action_np = controller.get_action() 
            action_t = torch.from_numpy(action_np).unsqueeze(0).unsqueeze(0).to(self.device).to(torch.bfloat16)
            
            z_next = self._solve_frame(action_t, self.current_z, self.t)
            with torch.no_grad():
                _, recon_frame = self.dec.forward_step(z_next, start_step_idx=self.t, update_cache=True)

            gen_frame = recon_frame[0, 0].float().cpu().permute(1, 2, 0).numpy()
            display_img = cv2.cvtColor(gen_frame, cv2.COLOR_RGB2BGR)
            display_img = np.clip(display_img * 255, 0, 255).astype(np.uint8)

            with self.frame_lock: self.latest_frame = display_img

            # --- STREAMING ADDITION ---
            if hasattr(controller, 'send_image'):
                controller.send_image(display_img)
            # --------------------------

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
    parser.add_argument("--context_len", type=int, default=10)
    parser.add_argument("--use_vr", action="store_true", help="Use VR Controller via UDP instead of Keyboard")

    args = parser.parse_args()

    device = get_device()
    
    print(f"Loading trajectories from {args.traj_path}...")
    traj_data = torch.load(args.traj_path, map_location="cpu")
    all_images = traj_data['images']
    all_actions = traj_data['actions']
    
    selector = TrajectorySelector(all_images)
    idx = selector.select()
    
    print(f"Initializing generator with trajectory {idx}...")
    selected_images = all_images[idx:idx+1].to(device).to(torch.bfloat16)
    selected_actions = all_actions[idx:idx+1].to(device).to(torch.bfloat16)

    generator = TrajectoryGenerator(args, device, selected_images, selected_actions)
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
            
            # Check for quit
            key = cv2.waitKey(10)
            if key == ord('q') or key == 27:
                print("Quit requested...")
                break
            
            if not generator.running:
                print("Generator finished naturally.")
                break
                
    except KeyboardInterrupt:
        print("KeyboardInterrupt received...")
    finally:
        # Order matters: stop threads first, then destroy windows
        print("Stopping Controller...")
        controller.stop()
        
        print("Stopping Generator...")
        generator.stop()
        
        print("Closing Windows...")
        cv2.destroyAllWindows()
        cv2.waitKey(1) # Small hack to ensure windows close on macOS
        print("Done.")

if __name__ == "__main__":
    main()
