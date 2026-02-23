import contextlib
import math
import time
import os
from functools import partial  # (kept in case you extend later)
import hydra
import cv2
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.nn.functional import interpolate
from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose
from dreamerv4.datasets import ShardedHDF5Dataset
from dreamerv4.utils.joy import XBoxController
from dreamerv4.models.utils import load_tokenizer
from dreamerv4.models.utils import load_denoiser
from dreamerv4.sampling import AutoRegressiveForwardDynamics
from dreamerv4.utils.joy import FingerSubscriber


# -------------------------------------------------------------------------
# Configurable paths / constants
# -------------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16
JOYSTICK_ID = 0
ACTION_SCALE = 0.1
NUM_INIT_FRAMES = 8
NUM_FORWARD_STEPS = 1000
CONTEXT_LEN=64

# -------------------------------------------------------------------------
# Initial latent extraction from a PushT episode
# -------------------------------------------------------------------------
def get_initial_latents_from_dataset(
    data_dir: str,
    resolution: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """
    Load one PushT episode, encode with tokenizer, and return an initial latent
    trajectory of length seq_len (or truncated).
    """
    dataset = ShardedHDF5Dataset(
        data_dir=data_dir,
        window_size=32,
        stride=1,
        split='train',
        train_fraction=0.9,
        split_seed=123,
        shuffle_windows=False
    )

    # Arbitrary index into that episode
    # batch = dataset[10]
    batch = dataset[500]
    imgs = batch["image"]  # (T, C, H, W)
    actions = batch["action"]  # (1, T, N_lat, D_lat)
    imgs = interpolate(imgs, resolution).to(device=device)[None] # resize to tokenizer resolution

    return imgs, actions[None]  # (1, T, N_lat, D_lat)

# -------------------------------------------------------------------------
# Joystick control loop
# -------------------------------------------------------------------------

@hydra.main(config_path="config", config_name="dynamics/finger-small", version_base=None)
def main(cfg: DictConfig):
    # 1. Load models
    print("Loading models")
    denoiser = load_denoiser(cfg, device, max_num_forward_steps=NUM_FORWARD_STEPS)
    tokenizer = load_tokenizer(cfg, device, max_num_forward_steps=NUM_FORWARD_STEPS)

    # 2. Initial latents from dataset
    print("Extracting initial latents from dataset...")
    # print(cfg.dataset.data_dir)
    imgs, actions = get_initial_latents_from_dataset(cfg.dataset.data_dir, resolution= tuple(cfg.dataset.resolution), device=device)

    world = AutoRegressiveForwardDynamics(denoiser, tokenizer, context_length=CONTEXT_LEN, device=device, dtype=dtype)
    
    use_cuda = (device.type == "cuda")
    ctx = torch.autocast("cuda", dtype=dtype) if use_cuda else contextlib.nullcontext()
    # Reset the world model with initial images and actions
    with ctx:
        world.reset(imgs[:,:NUM_INIT_FRAMES], actions[:, :NUM_INIT_FRAMES])
    
    action_t = torch.zeros(1, 1, actions.shape[-1]).to(device, dtype=dtype)
    joy = XBoxController(JOYSTICK_ID)
    # joy = FingerSubscriber("tcp://192.168.123.82:4242")
  
    print("Starting joystick-controlled DreamerV4 rollout...")
    for i in tqdm(range(NUM_FORWARD_STEPS-cfg.denoiser.context_length)):
        tic = time.time()
        states = joy.getState()
        # action_t[..., :] = torch.from_numpy(states).to(dtype=dtype, device=device)
        cmd_right = states["right_joy"] 
        cmd_left = states["left_joy"] 
        action_t[..., 0] = cmd_right[1]* 1.
        action_t[..., 1] = -cmd_right[0]* 1.
        action_t[..., 2] = cmd_left[1]*1.
        # action_t[..., 5] = cmd_left[0]*0.5
        # if states['right_bumper']:
        #     action_t[..., 6]=0.
        # else:
        #     action_t[..., 6]=1.
        # action_np = controller.get_action() 
        # if action_np[-1] == -1:
        #     action_np[-1] = 0
        
        with ctx:
            img = world.step(action_t)
                   
        img = (img[0].permute(1,2,0)*255).to(torch.uint8).cpu().numpy()
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (1024, 1024))
        cv2.imshow("dreamerv4", img)
        # controller.send_image(img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        while(time.time()-tic < 0.2):
            time.sleep(0.001)

    cv2.destroyAllWindows()

# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
