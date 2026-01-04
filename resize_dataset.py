#!/usr/bin/env python3
"""
Fast resize: episode-wise processing (low RAM), OpenCV for 10x faster resize.
No full shard load; slices/decompress only actual frames.
"""

import os
import argparse
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import cv2  # pip/conda install opencv-python


def resize_sharded_dataset(
    input_dir: str,
    output_dir: str,
    target_size: tuple = (128, 128),
    shard_start: int = 0,
    shard_end: int = None,
):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    metadata_file = input_path / 'metadata.json'
    import json
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    num_shards = metadata['num_shards']
    shard_end = min(shard_end or num_shards - 1, num_shards - 1)
    
    print(f"Processing shards {shard_start}-{shard_end}")
    
    for shard_idx in tqdm(range(shard_start, shard_end + 1), desc=f"Shards {shard_start}-{shard_end}"):
        shard_file = input_path / f"shard_{shard_idx:04d}.h5"
        if not shard_file.exists(): continue
        
        out_shard = output_path / f"shard_{shard_idx:04d}.h5"
        
        with h5py.File(shard_file, 'r') as fin, h5py.File(out_shard, 'w') as fout:
            episode_lengths = fin['episode_lengths'][:]
            num_episodes = len(episode_lengths)
            max_length = fin.attrs['max_length']
            action_shape = fin['actions'].shape[2:]
            
            # Infer channels from first non-empty
            channels = None
            for ep in range(num_episodes):
                if episode_lengths[ep] > 0:
                    sample_img = fin['images'][ep, 0]
                    channels = sample_img.shape[-1]
                    break
            if channels is None:
                print(f"Shard {shard_idx}: no data, skip")
                continue
            new_img_shape = (*target_size, channels)
            print(f"Shard {shard_idx}: channels={channels}, new_shape={new_img_shape}")
            
            # Create output datasets
            images_ds = fout.create_dataset(
                'images', (num_episodes, max_length, *new_img_shape),
                dtype=np.uint8, chunks=(1, max_length, *new_img_shape),
                compression='gzip', compression_opts=4
            )
            actions_ds = fout.create_dataset(
                'actions', (num_episodes, max_length, *action_shape),
                dtype=np.float32, chunks=(1, max_length, *action_shape),
                compression='gzip', compression_opts=4
            )
            fout.create_dataset('episode_lengths', data=episode_lengths)
            fout.attrs['num_episodes'] = num_episodes
            fout.attrs['max_length'] = max_length
            
            # Process episodes: slice actual data only, resize, pad & write row
            for ep in tqdm(range(num_episodes), desc=f"Shard {shard_idx} eps", leave=False):
                actual_len = episode_lengths[ep]
                if actual_len == 0: continue
                
                # Slice actual frames (small mem!)
                actual_images = fin['images'][ep, :actual_len]
                actual_actions = fin['actions'][ep, :actual_len]
                
                # Resize images (cv2 fast)
                resized_images = np.zeros((actual_len, *new_img_shape), dtype=np.uint8)
                for t in range(actual_len):
                    resized_images[t] = cv2.resize(
                        actual_images[t], target_size[::-1],  # (W,H)
                        interpolation=cv2.INTER_LANCZOS4
                    )
                
                # Pad & write full row
                pad_images = np.zeros((max_length, *new_img_shape), dtype=np.uint8)
                pad_actions = np.zeros((max_length, *action_shape), dtype=np.float32)
                pad_images[:actual_len] = resized_images
                pad_actions[:actual_len] = actual_actions
                images_ds[ep] = pad_images
                actions_ds[ep] = pad_actions
    
    print("Update metadata.json: 'image_shape': [128, 128, 3] (or printed channels).")


def main():
    parser = argparse.ArgumentParser(description="Fast parallel HDF5 resize")
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--target_size', type=int, nargs=2, default=[128, 128])
    parser.add_argument('--shard_start', type=int, default=0)
    parser.add_argument('--shard_end', type=int, default=None)
    args = parser.parse_args()
    resize_sharded_dataset(args.input_dir, args.output_dir, tuple(args.target_size),
                           args.shard_start, args.shard_end)


if __name__ == "__main__":
    main()
