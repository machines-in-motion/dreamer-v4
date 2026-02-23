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

def write_episode(episode_data, ep_length, output_path):
    with h5py.File(output_path, 'w') as f:
        comp_args = {'compression': 'gzip', 'compression_opts': 4}
        # Datasets
        dataset_handles = {}
        for key in episode_data:
            data = episode_data[key]
            if len(data) > 0:
                data = np.stack(data)
                assert data.shape[0]==ep_length, f'{key} field in the episode data should have the same number of frame as episode length'
                dataset_handles[key] = f.create_dataset(key, shape=(1, *data.shape), dtype=data.dtype, chunks=(1, min(data.shape[0], 128), *data.shape[1:]), **comp_args)
                dataset_handles[key][0] = data

        # For compatibility with SOAR
        f.create_dataset('episode_lengths', data=np.array([ep_length], dtype=np.int32))
        f.attrs['num_episodes'] = 1

def convert_dataset(args):
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading: {args.repo_id}")
    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.local_dir, video_backend='pyav', tolerance_s=0.05) if args.local_dir else LeRobotDataset(repo_id=args.repo_id, video_backend='pyav', tolerance_s=0.05)

    source_fps = dataset.fps
    target_fps = args.target_fps if args.target_fps else source_fps
    ds_ratio = max(1, int(source_fps // target_fps))
    
    print(f"FPS: {source_fps} -> {target_fps} (Subsample: {ds_ratio})")
    
    assert source_fps >= args.target_fps > 0, "Target FPS must be less than Source FPS and greater than 0"
    assert source_fps % args.target_fps == 0, "Source FPS must be divisible by Target FPS"
    
    # Episode intervals
    intervals = []
    for episode_info in dataset.meta.episodes:
        start = episode_info['dataset_from_index']
        end = episode_info['dataset_to_index']
        intervals.append((start, end))

    episode_counter = args.first_episode_number
    for start_idx, end_idx in intervals:
        start_idx = start_idx + args.trncate_length
        end_idx = end_idx - args.trncate_length

        prev_action = dataset[start_idx][args.action_key].clone()
        counter = 0
        current_episode = {'images': [],
                                'actions': [],
                                'actions_rel': [], 
                                'actions_abs': [], 
                                'efforts': [], 
                                'states': [],
                                'is_demo': []
                                }
        sample_counter = 0
        for idx in tqdm(range(start_idx, end_idx)):
            if counter % ds_ratio == 0:
                sample = dataset[idx]
                sample_counter+=1
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
                resized_img = resized_img.numpy().transpose(1, 2, 0).copy()
                resized_img = (resized_img*255).astype(np.uint8)
                resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
                if args.visualize:
                    cv2.imshow('vis', resized_img)
                    cv2.waitKey(1)
                
                current_episode['images'].append(resized_img)

                # Actions
                action = sample[args.action_key]
                action_rel = (action - prev_action)/(1./target_fps) # relative action is estimated absolute action velocity
                prev_action = action.clone()
                current_episode['actions_rel'].append(action_rel.clone().numpy())
                current_episode['actions_abs'].append(action.clone().numpy())
                
                if args.relative_actions:
                    current_episode['actions'].append(action_rel.numpy())
                else:
                    current_episode['actions'].append(action.numpy())

                # States
                if args.state_key is not None:
                    current_episode['states'].append(sample[args.state_key].numpy().copy())
                
                if args.effort_key is not None:
                    current_episode['efforts'].append(sample[args.effort_key].numpy().copy())

                # Is demo dataset
                if 'is_demo' not in sample:
                    is_demo = np.array([int(args.is_demo)])
                else:
                    is_demo = sample['is_demo'].numpy()
                    
                current_episode['is_demo'].append(is_demo)
            counter+=1 

        if sample_counter > 0:
            write_episode(current_episode,sample_counter, output_path/f'shard_{episode_counter:04d}.h5')
            episode_counter+=1  


    # Save Metadata
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump({
            'num_shards': len(intervals),
            'total_episodes': len(intervals),
            'image_shape': resized_img.shape,
            'action_shape': action.shape,
            'relative_actions': args.relative_actions,
        }, f, indent=2, default=str)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_id', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--local_dir', type=str, default=None)
    parser.add_argument('--first_episode_number', type=int, default=0) # shards name offset
    
    # Image Args
    parser.add_argument('--target_fps', type=int, default=10)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--trncate_length', type=int, default=0)
    parser.add_argument('--rectangular', action='store_true', default=False, help="Pad images to square")
    parser.add_argument('--visualize', action='store_true', default=False, help="Visualize the images as it's processe2000d")
    
    # Action/State Args
    parser.add_argument('--relative_actions', action='store_true', default=True, help="Compute Action - Prev_Action")
    parser.add_argument('--image_key', type=str, default='observation.images.camera_0')
    parser.add_argument('--state_key', type=str, default='observation.state')
    parser.add_argument('--effort_key', type=str, default='observation.effort')
    parser.add_argument('--action_key', type=str, default='action')
    parser.add_argument('--is_demo', type=bool, default=False)
    # State key removed as it is not needed for Action Deltas

    args = parser.parse_args()

    
    if LEROBOT_AVAILABLE: convert_dataset(args)
    else: print("Please install 'lerobot'")

if __name__ == "__main__":
    main()
