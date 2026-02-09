#!/usr/bin/env python3
"""
Convert Hugging Face / LeRobot datasets to sharded HDF5.
Features:
- Action-Relative Deltas (a_t - a_{t-1}) for Image-Only World Models
- Resizes images & Adjusts FPS
- Sharded HDF5 output
"""

import argparse
import os
import json
import numpy as np
import h5py
import torch
from pathlib import Path
from tqdm import tqdm
from torch.nn.functional import interpolate
import cv2

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    print("Warning: 'lerobot' library not found.")

def write_shard(episodes, output_path, shard_idx, image_shape, action_shape):
    if not episodes: return
    num_episodes = len(episodes)
    max_length = max(ep['length'] for ep in episodes)

    shard_filename = output_path / f"shard_{shard_idx:04d}.h5"
    with h5py.File(shard_filename, 'w') as f:
        comp_args = {'compression': 'gzip', 'compression_opts': 4}
        
        # Datasets
        images_ds = f.create_dataset('images', shape=(num_episodes, max_length, *image_shape), dtype=np.uint8, chunks=(1, 128, *image_shape), **comp_args)
        actions_ds = f.create_dataset('actions', shape=(num_episodes, max_length, *action_shape), dtype=np.float32, chunks=(1, 128, *action_shape), **comp_args)
        
        lengths = []
        for i, ep in enumerate(episodes):
            ep_len = ep['length']
            lengths.append(ep_len)
            
            # Images
            img_data = ep['images']
            if ep_len < max_length:
                padding = np.zeros((max_length - ep_len, *image_shape), dtype=np.uint8)
                img_data = np.concatenate([img_data, padding], axis=0)
            images_ds[i] = img_data
            
            # Actions
            act_data = ep['actions']
            if ep_len < max_length:
                padding = np.zeros((max_length - ep_len, *action_shape), dtype=np.float32)
                act_data = np.concatenate([act_data, padding], axis=0)
            actions_ds[i] = act_data
            
        f.create_dataset('episode_lengths', data=np.array(lengths, dtype=np.int32))
        f.attrs['num_episodes'] = num_episodes

def convert_dataset(args):
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading: {args.repo_id}")
    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.local_dir, video_backend='pyav', tolerance_s=0.05) if args.local_dir else LeRobotDataset(repo_id=args.repo_id, video_backend='pyav', tolerance_s=0.05)

    source_fps = dataset.fps
    target_fps = args.target_fps if args.target_fps else source_fps
    ds_ratio = max(1, int(source_fps // target_fps))
    
    print(f"FPS: {source_fps} -> {target_fps} (Subsample: {ds_ratio})")
    
    shard_buffer = []
    current_episode = {'images': [], 'actions': []}
    shard_idx = 0
    total_episodes = 0
    
    prev_episode_idx = dataset[0]['episode_index'].item()
    frame_idx_in_episode = 0
    
    # State tracking for Relative Actions
    # Initialize with first frame action to handle edge cases safely
    prev_action_val = dataset[0][args.action_key]
    
    sample_img_shape = None
    sample_act_shape = None
    assert source_fps > args.target_fps > 0, "Target FPS must be less than Source FPS and greater than 0"
    assert source_fps % args.target_fps == 0, "Source FPS must be divisible by Target FPS"
    for idx in tqdm(range(len(dataset))):
        sample = dataset[idx]
        curr_episode_idx = sample['episode_index'].item()
        
        # --- Episode Boundary ---
        if curr_episode_idx != prev_episode_idx:
            if len(current_episode['images']) > args.min_episode_length: # save the episode if it's long enough
                # Save Buffer
                # current_episode['images'] = current_episode['images'][:-1]
                # current_episode['actions'] = current_episode['actions'][1:]
                ep_imgs = np.stack(current_episode['images'])
                ep_acts = np.stack(current_episode['actions'])
                
                shard_buffer.append({'images': ep_imgs, 'actions': ep_acts, 'length': len(ep_imgs)})
                total_episodes += 1
                
                if len(shard_buffer) >= args.episodes_per_shard:
                    if sample_img_shape is None:
                        sample_img_shape = ep_imgs.shape[1:]
                        sample_act_shape = ep_acts.shape[1:]
                    write_shard(shard_buffer, output_path, shard_idx, sample_img_shape, sample_act_shape)
                    shard_buffer = []
                    shard_idx += 1
            
            current_episode = {'images': [], 'actions': []}
            prev_episode_idx = curr_episode_idx
            frame_idx_in_episode = 0
            # Reset prev_action is handled implicitly below by frame_idx check
        if frame_idx_in_episode == 0:
            prev_action_val = sample[args.action_key].clone() # Reset at episode start
        # --- Downsampling & Processing ---
        if frame_idx_in_episode % ds_ratio == 0:
            
            # 1. Image Processing
            img_tensor = sample[args.image_key] # C, H, W
            if args.rectangular:
                C, H, W = img_tensor.shape
                max_side = max(H, W)
                img_padded = torch.zeros((C, max_side, max_side), dtype=img_tensor.dtype)
                r, c = (max_side - H)//2, (max_side - W)//2
                img_padded[:, r:r+H, c:c+W] = img_tensor
                img_to_resize = img_padded
            else:
                img_to_resize = img_tensor

            if args.width and args.height:
                resized_img = interpolate(img_to_resize.unsqueeze(0), size=(args.height, args.width), mode='bilinear', align_corners=False).squeeze(0)
            else:
                resized_img = img_to_resize

            # 2. Relative Action Calculation (Previous-Action Relative)
            # Formula: a_delta = a_t - a_{t-1}
            # This is suitable for Image-Only World Models
            current_action_val = sample[args.action_key]
            
            if args.relative_actions:
                if frame_idx_in_episode == 0:
                    # First frame: No previous action, so delta is zero
                    action_rel = torch.zeros_like(current_action_val)
                else:
                    action_rel = current_action_val - prev_action_val
                
                # Update tracker for NEXT step
                prev_action_val = current_action_val.clone() # Clone is safer
            else:
                action_rel = current_action_val
            
            resized_img = resized_img.numpy().transpose(1, 2, 0).copy()
            resized_img = (resized_img*255).astype(np.uint8)
            resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            if args.visualize:
                cv2.imshow('vis', resized_img)
                cv2.waitKey(1)
            if frame_idx_in_episode > args.min_episode_length:
                current_episode['images'].append(resized_img)
                current_episode['actions'].append(action_rel.numpy())

        frame_idx_in_episode += 1

    # --- Flush Final ---
    if len(current_episode['images']) > 0:
        ep_imgs = np.stack(current_episode['images'])
        ep_acts = np.stack(current_episode['actions'])
        shard_buffer.append({'images': ep_imgs, 'actions': ep_acts, 'length': len(ep_imgs)})
        total_episodes += 1

    if len(shard_buffer) > 0:
        if sample_img_shape is None:
            sample_img_shape = shard_buffer[0]['images'].shape[1:]
            sample_act_shape = shard_buffer[0]['actions'].shape[1:]
        write_shard(shard_buffer, output_path, shard_idx, sample_img_shape, sample_act_shape)

    # Save Metadata
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump({
            'num_shards': shard_idx + (1 if len(shard_buffer)>0 else 0),
            'total_episodes': total_episodes,
            'image_shape': sample_img_shape if sample_img_shape else "N/A",
            'action_shape': sample_act_shape if sample_act_shape else "N/A",
            'relative_actions': args.relative_actions,
            'relative_mode': 'action_delta' if args.relative_actions else 'absolute'
        }, f, indent=2, default=str)
    
    print(f"\nDone. Processed {total_episodes} episodes.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_id', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--local_dir', type=str, default=None)
    parser.add_argument('--episodes_per_shard', type=int, default=1)
    
    # Image Args
    parser.add_argument('--target_fps', type=int, default=10)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--min_episode_length', type=int, default=1000)
    parser.add_argument('--rectangular', action='store_true', default=False, help="Pad images to square")
    parser.add_argument('--visualize', action='store_true', default=False, help="Visualize the images as it's processed")
    
    # Action/State Args
    parser.add_argument('--relative_actions', action='store_true', default=True, help="Compute Action - Prev_Action")
    parser.add_argument('--image_key', type=str, default='observation.images.camera_0')
    parser.add_argument('--action_key', type=str, default='action')
    # State key removed as it is not needed for Action Deltas

    args = parser.parse_args()

    
    if LEROBOT_AVAILABLE: convert_dataset(args)
    else: print("Please install 'lerobot'")

if __name__ == "__main__":
    main()
