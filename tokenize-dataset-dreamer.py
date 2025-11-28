import hydra
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SingleViewSequenceDataset
from model.tokenizer import (CausalTokenizerDecoder, 
                             CausalTokenizerEncoder, 
                             CausalTokenizerConfig, 
                             TokensToImageHead, 
                             ImagePatchifier)
from model.utils import TokenMasker
from pathlib import Path
import h5py

def load_checkpoint(ckpt_path, model, optimizer=None, scaler=None, scheduler=None):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    model.load_state_dict(ckpt["model"])

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])

    step = ckpt.get("step", 0)
    epoch = ckpt.get("epoch", 0)

    print(f"🔄 Loaded checkpoint from {ckpt_path} at step={step}, epoch={epoch}")
    return step, epoch

class ModelWrapper(nn.Module):
    def __init__(self, cfg:DictConfig):
        super().__init__()
        self.cfg = cfg
        tokenizer_cfg = CausalTokenizerConfig(**OmegaConf.to_object(cfg.tokenizer)) 
        self.encoder = CausalTokenizerEncoder(tokenizer_cfg)
        self.decoder = CausalTokenizerDecoder(tokenizer_cfg)
        self.patchifier = ImagePatchifier(cfg.tokenizer.patch_size, cfg.tokenizer.model_dim)
        self.image_head = TokensToImageHead(cfg.tokenizer.model_dim, cfg.dataset.resolution, cfg.tokenizer.patch_size)
        self.masker = TokenMasker(cfg.tokenizer.model_dim, cfg.tokenizer.num_modality_tokens)

    def forward(self, images):
        images = (images*2.)-1. # Translate the images in +-1 range
        tokens = self.patchifier(images)
        masked_tokens = self.masker(tokens)
        z, _ = self.encoder(masked_tokens)
        z_decoded = self.decoder(z)
        recon_images = self.image_head(z_decoded)
        return  torch.clamp((recon_images + 1)/2., 0., 1.)
    
    def tokenize(self, images):
        images = (images*2.)-1. # Translate the images in +-1 range
        tokens = self.patchifier(images)
        z, _ = self.encoder(tokens)
        return z

@hydra.main(config_path="config", config_name="tokenizer_small.yaml")
def main(cfg: DictConfig):
    torch.set_float32_matmul_precision('high')
    device = cfg.train.device
    # Instantiate the model
    model = ModelWrapper(cfg)
    if cfg.reload_checkpoint:
        load_checkpoint(cfg.reload_checkpoint, model)
    model.to(device)
    model = torch.compile(model, mode=cfg.train.torch_compile_mode)
    model.eval()
    # Dataset and data loaders
    dataset_base_path = Path(cfg.dataset.base_path)
    for seq_file in cfg.dataset.train_sequences:
        print(f'Processing seq: {seq_file}')
        dataset = SingleViewSequenceDataset(dataset_base_path/seq_file, cfg.tokenizer.max_context_length, load_to_ram=cfg.dataset.load_to_ram)
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=cfg.train.num_workers,
            pin_memory=True,
            persistent_workers=True,
            shuffle=True,
        )
        tokens = []
        for batch in tqdm(dataloader):
            ctx_len = -1
            imgs = batch['observation.image'][:,:ctx_len, ... ].to(torch.float32).to(device)
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    token = model.tokenize(imgs).to(torch.float32).cpu()
                    tokens.append(token.clone())
        tokens = torch.concat(tokens, dim=0)
        dataset.close() # Close the HD5 dataset before writing into it
        with h5py.File(dataset_base_path/seq_file, 'a') as f:
            if "dreamer-tokens" in f:
                del f["dreamer-tokens"]
            dataset = f.create_dataset("dreamer-tokens", data=tokens.numpy(), chunks=(4, cfg.tokenizer.max_context_length-1, cfg.tokenizer.num_latent_tokens, cfg.tokenizer.latent_dim), compression="gzip")

if __name__ == '__main__':
    main()
