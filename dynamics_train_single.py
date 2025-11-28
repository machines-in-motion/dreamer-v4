import time
from PIL import Image
import hydra
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import ConcatDataset
import kornia.augmentation as K
from dataset import SingleViewSequenceDataset
from model.tokenizer import (CausalTokenizerDecoder, 
                             CausalTokenizerEncoder, 
                             CausalTokenizerConfig, 
                             TokensToImageHead, 
                             ImagePatchifier)
from model.utils import TokenMasker
import torch.optim as optim
import lpips
from pathlib import Path
import wandb
from transformers import get_cosine_schedule_with_warmup

class RMSLossScaler:
    """
    Tracks running RMS for named losses and returns normalized losses.
    """
    def __init__(self, decay: float = 0.99, eps: float = 1e-8):
        self.decay = decay
        self.eps = eps
        self.ema_sq = {}  # name -> scalar tensor

    def __call__(self, name: str, value: torch.Tensor) -> torch.Tensor:
        # value is a scalar loss tensor (per-batch, per-rank)
        with torch.no_grad():
            sq = value.detach().pow(2)
            mean_sq = sq.mean()

            if name not in self.ema_sq:
                self.ema_sq[name] = mean_sq
            else:
                self.ema_sq[name] = (
                    self.decay * self.ema_sq[name] + (1.0 - self.decay) * mean_sq
                )

            rms = (self.ema_sq[name] + self.eps).sqrt()

        # Normalize current loss by running RMS; gradient flows only through value
        return value / rms

def save_checkpoint(model, optimizer, step, epoch, cfg, scaler=None, scheduler=None):
    ckpt_dir = Path(cfg.output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"checkpoint_step_{step:07d}.pt"

    # Fix: Access _orig_mod for compiled models
    model_state = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()

    checkpoint = {
        "model": model_state,  # ← CHANGED
        "optimizer": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "cfg": OmegaConf.to_container(cfg, resolve=True),
    }

    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()

    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    torch.save(checkpoint, ckpt_path)
    print(f"🔒 Checkpoint saved at: {str(ckpt_path)}")

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

@hydra.main(config_path="config", config_name="tokenizer_small.yaml")
def main(cfg: DictConfig):
    # --------------------------------------------------------------------
    #  W&B SETUP
    # --------------------------------------------------------------------
    # wandb.init(project="dreamerv4-tokenizer")
    if cfg.wandb.enable:
        if cfg.wandb.api_key:
            wandb.login(key=cfg.wandb.api_key)
        else:
            wandb.login()   # rely on env var or already logged-in session
        wandb_config = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(
            project=cfg.wandb.project,
            entity="rk4342",
            config=wandb_config
        )
    # Torch configs
    if cfg.train.enable_fast_matmul:
        torch.set_float32_matmul_precision('high')
    device = cfg.train.device

    # Dataset and data loaders
    dataset_base_path = Path(cfg.dataset.base_path)
    train_datasets = [SingleViewSequenceDataset(dataset_base_path/seq_file, cfg.tokenizer.max_context_length, load_to_ram=cfg.dataset.load_to_ram)
            for seq_file in cfg.dataset.train_sequences]
    train_dataset = ConcatDataset(train_datasets)

    test_datasets = [SingleViewSequenceDataset(dataset_base_path/seq_file, cfg.tokenizer.max_context_length, load_to_ram=cfg.dataset.load_to_ram)
            for seq_file in cfg.dataset.test_sequences]
    test_dataset = ConcatDataset(test_datasets)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        persistent_workers=True,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        persistent_workers=True,
        shuffle=True,
    )
    test_iter = iter(test_loader)
    if cfg.augmentation.enable:
        augmentor = torch.nn.Sequential(
            K.RandomGaussianBlur(tuple(cfg.augmentation.gaussian_blur.kernel_size),
                                 tuple(cfg.augmentation.gaussian_blur.sigma_range), 
                                 p=cfg.augmentation.gaussian_blur.application_probability),
            
            K.ColorJitter(brightness=cfg.augmentation.color_jitter.brightness, 
                          contrast=  cfg.augmentation.color_jitter.contrast, 
                          saturation=cfg.augmentation.color_jitter.saturation,
                          hue=cfg.augmentation.color_jitter.hue,
                          p=cfg.augmentation.color_jitter.application_probability),
            
            K.RandomGaussianNoise(mean=cfg.augmentation.random_noise.mean, 
                                  std=cfg.augmentation.random_noise.std)
        )

    # Instantiate the model
    model = ModelWrapper(cfg)
    model.to(device)
    # print('Compiling the model...')
    model = torch.compile(model, mode=cfg.train.torch_compile_mode)
    # model = torch.compile(model, mode="max-autotune", fullgraph=False)

    num_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    print(f"Number of encoder parameters (M): {num_params/1e6:.2f}M")
    num_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    print(f"Number of decoder parameters (M): {num_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    N_steps = cfg.train.num_epochs*len(train_loader)
    WARMUP_STEPS = N_steps * 0.02
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=N_steps,
        num_cycles=0.5
    )
    # RMS loss normalizer for tokenizer losses
    rms_norm = RMSLossScaler(decay=0.99, eps=1e-8)
    mse_loss_fn = nn.MSELoss()
    lpips_model = lpips.LPIPS(net='vgg').to(device=device, dtype=torch.bfloat16)
    lpips_model.eval()
    for p in lpips_model.parameters():
        p.requires_grad_(False)

    step = 0
    print('Training started:')
    for epoch in range(cfg.train.num_epochs):
        for batch in train_loader:
            step += 1
            # ctx_len = torch.randint(1, cfg.tokenizer.max_context_length,(1,)).item()
            ctx_len = -1
            imgs = batch['observation.image'][:,:ctx_len, ... ].to(torch.float32).to(device)
            if cfg.augmentation.enable:
                B, T, C, H, W = imgs.shape
                imgs = imgs.view(B*T, C, H, W).contiguous().to(device)      # flatten time
                imgs = augmentor(imgs) 
                imgs = imgs.view(B, T, C, H, W).to(torch.float32)     # restore shape
            
            torch.compiler.cudagraph_mark_step_begin()
            optimizer.zero_grad()
            model.train()

            if cfg.train.mixed_precision:           
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    recon_images = model(imgs)
                    mse = mse_loss_fn(recon_images, imgs)
            else:
                recon_images = model(imgs)
                mse = mse_loss_fn(recon_images, imgs)

            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    images_lpips = (imgs * 2.0) - 1.0        
                    x_hat_lpips = (recon_images * 2.0) - 1.0
                    B, T, C, H, W = images_lpips.shape
                    images_lpips_flat = images_lpips.view(B * T, C, H, W)
                    x_hat_lpips_flat = x_hat_lpips.view(B * T, C, H, W)
                    lpips_val = lpips_model(x_hat_lpips_flat, images_lpips_flat)
                    lpips_loss = lpips_val.mean()

            raw_loss = mse + 0.2 * lpips_loss
            mse_norm = rms_norm("mse", mse)
            lpips_norm = rms_norm("lpips", lpips_loss)
            loss = (mse_norm + 0.2 * lpips_norm)
            loss.backward()
            
            if cfg.train.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_grad_norm)

            optimizer.step()
            scheduler.step()
            if step % cfg.print_every==0:
                print(f"RMSE Loss: {loss.item()}, Raw_Loss: {raw_loss.item()}")
                # wandb.log({
                #     "epoch": epoch,
                #     "training_loss": total_loss,
                #     "lr": optimizer.param_groups[0]['lr'],
                #     # "best loss": best_loss
                # })

            # ------------------------------------------------------------------
            #  LOG LOSS TO WANDB
            # ------------------------------------------------------------------
            if cfg.wandb.enable and (step % cfg.train.log_every == 0):
                wandb.log({
                    "train/mse_loss": mse.item(),
                }, step=step)


            if step % cfg.plot_every == 0:
                # ------------------------------------------------------------------
                #  EVALUATION STEP ON RANDOM TEST BATCH
                # ------------------------------------------------------------------
                model.eval()
                with torch.no_grad():
                    # sample a random batch
                    try:
                        test_batch = next(test_iter)
                    except:
                        test_iter = iter(test_loader)
                        test_batch = next(test_iter)

                    test_imgs = test_batch['observation.image'].to(torch.float32).to(device)#[:,:16,...]
                    B, T, C, H, W = test_imgs.shape

                    recon = model(test_imgs)               # (B, T, C, H, W)
                    recon = recon.clamp(0.0, 1.0)

                    # --------------------------------------------------------------
                    #  BUILD TILE: odd rows = recon, even rows = GT
                    #  Tile size: (2B rows, T columns)
                    # --------------------------------------------------------------
                    rows = []
                    for b in range(B):
                        row_recon = torch.cat([recon[b, t] for t in range(T)], dim=2)  # concat along width
                        row_gt    = torch.cat([test_imgs[b, t] for t in range(T)], dim=2)
                        rows.append(row_recon)
                        rows.append(row_gt)
                    tiled = torch.cat(rows, dim=1)   # stack vertically

                    # --------------------------------------------------------------
                    #  Convert to PIL Image
                    # --------------------------------------------------------------
                    tiled_np = (tiled.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')

                    tiled_img = Image.fromarray(tiled_np)

                    # --------------------------------------------------------------
                    #  Resize to a W&B-friendly size (max 512px)
                    # --------------------------------------------------------------
                    max_dim = 4096
                    w, h = tiled_img.size
                    scale = min(max_dim / max(w, h), 1.0)
                    new_w, new_h = int(w * scale), int(h * scale)
                    tiled_img_resized = tiled_img.resize((new_w, new_h), Image.BILINEAR)

                    # --------------------------------------------------------------
                    #  Save image locally
                    # --------------------------------------------------------------
                    output_dir = Path(cfg.output_dir) / "plots"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    file_path = output_dir / f"step_{step:06d}.png"
                    tiled_img_resized.save(file_path)

                    # --------------------------------------------------------------
                    #  Optional W&B logging
                    # --------------------------------------------------------------
                    if cfg.wandb.enable:
                        wandb.log({"reconstruction_preview": wandb.Image(tiled_img_resized)}, step=step)
        save_checkpoint(model, optimizer, step, epoch, cfg)

if __name__ == '__main__':
    main()
