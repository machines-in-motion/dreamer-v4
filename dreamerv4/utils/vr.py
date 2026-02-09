import cv2
import struct
import shutil
import subprocess
import socket
import threading
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R

class GstServer:
    def __init__(self, port=5000, width=256, height=256, fps=24):
        if not shutil.which("gst-launch-1.0"):
            print("Error: gst-launch-1.0 not found. Install GStreamer.")
            self.proc = None
            return

        # Pipeline: Raw BGR -> x264 (Low Latency) -> TCP Server
        # 'queue leaky=downstream' ensures the Python script never blocks if no client connects.
        self.cmd = (
            f"gst-launch-1.0 -q fdsrc ! "
            f"videoparse format=bgr width={width} height={height} framerate={fps}/1 ! "
            "videoconvert ! "
            "x264enc tune=zerolatency speed-preset=ultrafast key-int-max=30 bitrate=500 ! "
            "video/x-h264,stream-format=byte-stream ! "
            "h264parse config-interval=1 ! "
            "queue leaky=downstream max-size-buffers=1 ! "
            f"tcpserversink host=0.0.0.0 port={port} sync-method=latest-keyframe"
        )
        
        print(f"Starting GStreamer TCP Server on port {port}...")
        self.proc = subprocess.Popen(self.cmd.split(), stdin=subprocess.PIPE)

    def write(self, frame):
        """Frame must be a (H, W, 3) uint8 numpy array (BGR)"""
        if self.proc and self.proc.stdin:
            try:
                self.proc.stdin.write(frame.tobytes())
                self.proc.stdin.flush()
            except BrokenPipeError:
                # GStreamer process crashed or closed
                print("GStreamer pipe broken.")
                self.close()

    def close(self):
        if self.proc:
            if self.proc.stdin: self.proc.stdin.close()
            self.proc.terminate()
            self.proc.wait()
            self.proc = None

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
        self.frame = None
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
        # if controller.frame is not None:
        #     w_T_right = controller.frame.right_pose.matrix
        #     delta_pose = np.linalg.inv(w_T_right_prev)@w_T_right
        #     w_T_right_prev = w_T_right.copy()
        #     vel_cmd = delta_pose[:3, -1]
        #     omega_cmd = R.from_matrix(delta_pose[:3, :3]).as_rotvec()
        #     run = controller.frame.right_input.hand_trigger > 0.5
        # if run:
        #     action_np[0] = vel_cmd[0]
        #     action_np[1] = 0#vel_cmd[1]
        #     action_np[2] = 0#vel_cmd[2]
        #     action_np[5] = -omega_cmd[1]
        #     action_np[4] = omega_cmd[0]
        #     action_np[3] = -omega_cmd[2]
        # action_t[...,:] = torch.tensor(action_np).to(action_t.device, dtype=action_t.dtype)
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
                self.frame = frame
                
                # Placeholder for Rotation (if you want to implement rotation deltas later)
                # Extracting Euler angles from 4x4 matrix is slightly complex, 
                # usually involves `scipy.spatial.transform.Rotation`
                curr_rot = np.zeros(3) 

                # Get Head Orientation (Forward and Right vectors)
                # Unity View Matrix (localToWorld): Column 2 is Forward (Z), Column 0 is Right (X)
                # We flatten them to the ground (y=0) to ignore looking up/down
                head_fwd = frame.head_pose.matrix[:3, 2] # Z axis (Forward)
                head_fwd[1] = 0 # Flatten to ground
                head_fwd = head_fwd / (np.linalg.norm(head_fwd) + 1e-6) # Normalize

                head_right = frame.head_pose.matrix[:3, 0] # X axis (Right)
                head_right[1] = 0 # Flatten
                head_right = head_right / (np.linalg.norm(head_right) + 1e-6)
                if frame.right_input.hand_trigger > 0.5:
                    with self.lock:
                        if self.last_pos is not None:
                            delta_world = curr_pos - self.last_pos
                            
                            # 3. Project Delta onto Head Axes (Dot Product)
                            # This gives movement relative to where you are looking
                            forward_change = np.dot(delta_world, head_fwd)
                            right_change = np.dot(delta_world, head_right)
                            vertical_change = delta_world[1] # Unity Y is Up

                            # 4. Map to Action (Adjust signs to match your robot)
                            # Action 0: Forward (Robot X)
                            self._current_action[0] += forward_change 
                            
                            # Action 1: Left/Right (Robot Y). 
                            # If Robot +Y is Left, use positive. If Right, use negative.
                            # Your original code had "-delta_pos[0]" for X (Right), so assuming Action 1 is Left.
                            self._current_action[1] -= right_change 
                            
                            # Action 2: Up/Down (Robot Z)
                            self._current_action[2] += vertical_change

                            self._current_action = np.clip(self._current_action, -self.max_delta, self.max_delta)
                            self._current_action[5] += frame.left_input.joystick[0]
                        
                        # Update Gripper State
                        # Using Right Index Trigger (0.0 to 1.0)
                        # Map 0.0 -> -1 (Closed), 1.0 -> 1 (Open) ?? 
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

    def stop(self):        # print(action_np)

        super().stop()
        self.sock.close()



