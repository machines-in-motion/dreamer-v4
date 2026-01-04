import argparse
import torch
import numpy as np
from dataset import ShardedHDF5Dataset, HDF5SequenceDataset

def export_trajectory(args):
    print(f"Loading dataset from {args.data_dir}...")
    
    """dataset = ShardedHDF5Dataset(
        data_dir=args.data_dir, 
        window_size=args.window_size, 
        stride=1,
        split="test",
        train_fraction=0.9, split_seed=123
    )"""
    """dataset = ShardedHDF5Dataset(
        data_dir=args.data_dir, 
        window_size=args.window_size, 
        stride=1,
        split="train",
        train_fraction=0.9, split_seed=123
    )"""
    dataset = HDF5SequenceDataset(
        data_dir=args.data_dir,
        window_size=args.window_size,
        stride=1,
    )
    
    total_items = len(dataset)
    num_to_sample = min(args.num_trajs, total_items)
    
    print(f"Selecting {num_to_sample} random trajectories from {total_items} available...")
    
    # Randomly sample indices
    indices = np.random.choice(total_items, num_to_sample, replace=False)
    
    batch_images = []
    batch_actions = []
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        # sample['image'] is [T, C, H, W]
        # sample['action'] is [T, D]
        batch_images.append(sample['image'])
        batch_actions.append(sample['action'])
        
        if (i + 1) % 5 == 0:
            print(f"  Extracted {i + 1}/{num_to_sample}...")

    # Stack into [B, T, C, H, W]
    saved_images = torch.stack(batch_images)
    saved_actions = torch.stack(batch_actions)

    print(f"Final Shapes:")
    print(f"  Images: {saved_images.shape}")
    print(f"  Actions: {saved_actions.shape}")
    
    # Save to disk
    torch.save({
        'images': saved_images.clone(),
        'actions': saved_actions.clone()
    }, args.output_path)
    
    print(f"Saved batch successfully to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="trajs_32.pt")
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--num_trajs", type=int, default=32, help="Number of random trajectories to extract")
    
    args = parser.parse_args()
    export_trajectory(args)
