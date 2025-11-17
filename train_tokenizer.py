
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from model.tokenizer import (CausalTokenizerDecoder, 
                             CausalTokenizerEncoder, 
                             CausalTokenizerConfig, 
                             TokensToImageHead, 
                             ImagePatchifier)
from model.utils import TokenMasker
import torch.optim as optim
import lpips

torch.set_float32_matmul_precision('high')
# torch.backends.cuda.enable_flash_sdp(True)
# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_math_sdp(False)

device = 'cuda:0'

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
        tokens = self.patchifier(images)
        masked_tokens = self.masker(tokens)
        z, _ = self.encoder(masked_tokens)
        z_decoded = self.decoder(z)
        recon_images = self.image_head(z_decoded)
        return recon_images

@hydra.main(config_path="config", config_name="tokenizer.yaml")
def main(cfg: DictConfig):
    model = ModelWrapper(cfg)
    model.to(device)
    print('Compiling the model...')
    # model = torch.compile(model, mode="max-autotune")
    model = torch.compile(model)
    num_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    print(f"Number of encoder parameters (M): {num_params/1e6:.2f}M")
    num_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    print(f"Number of decoder parameters (M): {num_params/1e6:.2f}M")

    mse_loss_fn = nn.MSELoss()
    # lpips_loss_fn = lpips.LPIPS(net='vgg').eval().to(device)   # perceptual loss

    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)

    B = cfg.train.batch_size
    C = 3
    H = cfg.dataset.resolution[0]
    W = cfg.dataset.resolution[1]
    T = 96
    images = torch.randn(B, T, C, H, W).to(device)
    for step in range(int(cfg.train.num_training_steps)):
        optimizer.zero_grad()
        for _ in range(cfg.train.accum_grad_steps):
            tic = time.time()
            
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                recon_images = model(images)
                mse = mse_loss_fn(recon_images, images)
            
            print(f'forward computation time: {time.time()-tic}')
            #lp=lpips_loss_fn(recon_images.flatten(0,1), images.flatten(0,1)).mean()
            #total_loss = mse + cfg.train.lpips_weight * lp
            total_loss = mse
            total_loss.backward()
            print(f'backward computation time: {time.time()-tic}')


        optimizer.step()
        print(total_loss.item())

if __name__ == '__main__':
    main()
