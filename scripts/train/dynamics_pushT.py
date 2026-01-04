import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
import kornia.augmentation as K
from dataset import SingleViewSequenceDataset
from model.dynamics import *
from pathlib import Path
from transformers import get_cosine_schedule_with_warmup
import wandb  # NEW


def save_checkpoint(model, optimizer, step, epoch, cfg, scaler=None, scheduler=None):
    ckpt_dir = Path(cfg.output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"checkpoint_step_{step:07d}.pt"

    # Fix: Access _orig_mod for compiled models
    model_state = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()

    checkpoint = {
        "model": model_state,
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


@hydra.main(config_path="config", config_name="dynamics_small.yaml")
def main(cfg: DictConfig):
    # Torch configs
    if cfg.train.enable_fast_matmul:
        torch.set_float32_matmul_precision("high")

    # FIX: use torch.device and derive device_type for autocast
    device = torch.device(cfg.train.device)
    device_type = "cuda" if device.type == "cuda" else "cpu"

    # Dataset and data loaders
    dataset_base_path = Path(cfg.dataset.base_path)

    train_datasets = [
        SingleViewSequenceDataset(
            dataset_base_path / seq_file,
            cfg.train.short_context,
            load_to_ram=cfg.dataset.load_to_ram,
        )
        for seq_file in cfg.dataset.train_sequences
    ]
    train_dataset = ConcatDataset(train_datasets)

    test_datasets = [
        SingleViewSequenceDataset(
            dataset_base_path / seq_file,
            cfg.train.short_context,
            load_to_ram=cfg.dataset.load_to_ram,
        )
        for seq_file in cfg.dataset.test_sequences
    ]
    test_dataset = ConcatDataset(test_datasets)

    persistent = cfg.train.num_workers > 0  # FIX

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        shuffle=True,
    )

    # Instantiate the model
    denoiser_cfg = DreamerV4DenoiserCfg(**OmegaConf.to_object(cfg.denoiser))
    model = DreamerV4Denoiser(denoiser_cfg)
    forward_diffuser = ForwardDiffusionWithShortcut(K_max=cfg.denoiser.K_max)

    model.to(device)
    model = torch.compile(model, mode="max-autotune", fullgraph=False)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of model parameters (M): {num_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    N_steps = cfg.train.num_epochs * len(train_loader)
    WARMUP_STEPS = int(N_steps * 0.02)  # FIX: cast to int

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=N_steps,
        num_cycles=0.5,
    )

    rms_norm = RMSLossScaler(decay=0.99, eps=1e-8)

    # NEW: wandb init (if enabled in config)
    use_wandb = "wandb" in cfg and getattr(cfg.wandb, "enable", False)
    if use_wandb:
        wandb.init(
            project=cfg.wandb.project,
            name=getattr(cfg.wandb, "run_name", None),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.watch(model, log="all", log_freq=100)

    step = 0
    print("Training started:")
    for epoch in range(cfg.train.num_epochs):
        for batch in train_loader:
            torch.compiler.cudagraph_mark_step_begin()

            optimizer.zero_grad(set_to_none=True)
            model.train()

            latent_tokens = batch["observation.tokens"].to(device, non_blocking=True)  # B x T x N x D
            actions = batch["action"].to(device, non_blocking=True)  # B x T x n_u (unused for now)


            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                latent_tokens_bf16 = latent_tokens.to(torch.bfloat16)
                info = forward_diffuser(latent_tokens_bf16)
                flow_loss, bootstrap_loss = compute_bootstrap_diffusion_loss(info, model, continuous_actions=actions)
                loss_raw = flow_loss + bootstrap_loss
                loss = rms_norm("loss", loss_raw)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_grad_norm)
            optimizer.step()
            scheduler.step()

            step += 1  # FIX

            # Logging
            if step % cfg.print_every == 0:
                print(
                    f"[epoch {epoch} | step {step}] "
                    f"Flow Loss: {flow_loss.item():.4f}, "
                    f"Bootstrap Loss: {bootstrap_loss.item():.4f}, "
                    f"RMS Loss: {loss.item():.4f}, "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )

            if use_wandb:
                wandb.log(
                    {
                        "train/flow_loss": flow_loss.item(),
                        "train/bootstrap_loss": bootstrap_loss.item(),
                        "train/loss_raw": loss_raw.item(),
                        "train/loss_rms": loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                    },
                    step=step,
                )

        save_checkpoint(model, optimizer, step, epoch, cfg, scheduler=scheduler)

        # (Optional) quick sanity eval on test_loader could be added here and logged to wandb

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()