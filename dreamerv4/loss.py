
def ramp_weight(tau: torch.Tensor) -> torch.Tensor:
    """
    Eq. (8): w(τ) = 0.9 τ + 0.1

    tau: (B, T) or (B, T, 1, 1)
    returns: same shape as tau
    """
    return 0.9 * tau + 0.1


# ------------------------------------------------------------
# 5. Forward diffusion with per-frame (τ_t, d_t)
# ------------------------------------------------------------

class ForwardDiffusionWithShortcut(nn.Module):
    """
    Dyadic shortcut schedule with per-frame (τ_t, d_t).

    K_max must be a power of 2, e.g. 32 or 64.
    Finest step size: d_min = 1 / K_max.

    Index convention used:
      max_pow2 = log2(K_max)
      step_index_raw ∈ {0..max_pow2}  (defines d = 1 / 2^step_index_raw)
      step_index      = max_pow2 - step_index_raw
        -> step_index = 0       ↔ d = d_min
        -> step_index = max_pow2 ↔ d = 1
    """

    def __init__(self, K_max=32):
        super().__init__()
        assert (K_max & (K_max - 1)) == 0, "K_max must be a power of 2"
        self.K_max = int(K_max)
        self.max_pow2 = int(math.log2(self.K_max))      # e.g. K_max=32 -> max_pow2=5
        self.d_min = 1.0 / float(self.K_max)            # finest step

    def sample_step_noise(self, batch_size, seq_len, device):
        """
        Returns per-frame diffusion parameters:

          tau             : (B, T) float   τ_t ∈ {0, d_t, 2 d_t, ..., 1 - d_t}
          step            : (B, T) float   d_t ∈ {1, 1/2, ..., 1/K_max}
          tau_index       : (B, T) long    τ index on finest grid: τ_t = tau_index * d_min
          step_index      : (B, T) long    our index in [0..max_pow2]; 0 ↔ d_min, max_pow2 ↔ 1
          half_step_index : (B, T) long    index for d_t/2 in same convention (step_index - 1)
          tau_plus_half_index: (B, T) long index on finest grid for τ_t + d_t/2
        """
        B, T = batch_size, seq_len

        # step_index_raw ∈ {0, ..., max_pow2}, defines d_t = 1 / 2^{step_index_raw}
        step_index_raw = torch.randint(
            low=0,
            high=self.max_pow2 + 1,
            size=(B, T),
            device=device,
            dtype=torch.long,
        )  # (B, T)

        # d_t = 1 / 2^{step_index_raw}
        step = 1.0 / (2.0 ** step_index_raw.float())     # (B, T) float

        # Our convention: step_index = max_pow2 - step_index_raw
        # step_index = 0  ↔ d = d_min,  step_index = max_pow2 ↔ d = 1
        step_index = self.max_pow2 - step_index_raw      # (B, T) long in [0..max_pow2]

        # Number of τ levels for this d_t: 1 / d_t = 2^{step_index_raw}
        num_levels = (2 ** step_index_raw).float()       # (B, T) float

        # m_t ∈ {0, ..., 2^{step_index_raw} - 1}
        m = torch.floor(
            torch.rand(B, T, device=device) * 0.9999 * num_levels
        ).long()                                         # (B, T) long

        tau = m.float() * step                           # (B, T) float, τ_t ∈ {0, d_t, ..., 1-d_t}

        # τ index on the finest grid of size K_max:
        # τ_t = m * d_t; d_t / d_min = 2^{step_index}; so τ_t / d_min = m * 2^{step_index}
        tau_index = m * (2 ** step_index)                # (B, T) long in [0 .. K_max-1]

        # stride on finest grid for step d_t: d_t / d_min = 2^{step_index}
        stride_full = (2 ** step_index)                  # (B, T) long
        delta_tau_index = stride_full // 2               # corresponds to d_t/2

        tau_plus_half_index = tau_index + delta_tau_index     # (B, T) long
        tau_plus_half_index = torch.clamp(
            tau_plus_half_index, min=0, max=self.K_max - 1
        )

        # half-step index in our convention: step_index - 1
        half_step_index = torch.clamp(step_index - 1, min=0)  # (B, T) long

        return dict(
            tau=tau,                                      # (B, T) float
            step=step,                                    # (B, T) float
            tau_index=tau_index,                          # (B, T) long
            step_index=step_index,                        # (B, T) long
            half_step_index=half_step_index,              # (B, T) long
            tau_plus_half_index=tau_plus_half_index,      # (B, T) long
        )

    def forward(self, z_clean: torch.Tensor):
        """
        z_clean: (B, T, N_lat, D_latent)  # clean latents (x = z_1)
        """
        B, T, N_lat, D_lat = z_clean.shape
        device = z_clean.device

        # Sample z_0 ~ N(0, I) in latent space
        z0 = torch.randn_like(z_clean)                      # (B, T, N_lat, D_lat)

        diff = self.sample_step_noise(B, T, device)

        tau = diff["tau"].unsqueeze(-1).unsqueeze(-1)       # (B, T, 1, 1)
        step = diff["step"].unsqueeze(-1).unsqueeze(-1)     # (B, T, 1, 1)

        # z_τ = (1 - τ) z_0 + τ z_1
        z_tau = (1.0 - tau) * z0 + tau * z_clean            # (B, T, N_lat, D_lat)

        return {
            "x": z_clean,                          # (B, T, N_lat, D_lat)  clean z_1
            "x_tau": z_tau,                        # (B, T, N_lat, D_lat)  noisy z_τ
            "tau": diff["tau"],                    # (B, T) float
            "step": diff["step"],                  # (B, T) float
            "tau_index": diff["tau_index"],        # (B, T) long
            "step_index": diff["step_index"],      # (B, T) long, 0 ↔ d_min
            "half_step_index": diff["half_step_index"],            # (B, T) long
            "tau_plus_half_index": diff["tau_plus_half_index"],    # (B, T) long
            "d_min": self.d_min,                   # scalar float
            "K_max": self.K_max,
        }


# ------------------------------------------------------------
# 6. apply_denoiser helper
# ------------------------------------------------------------

def apply_denoiser(
    denoiser: DreamerV4Denoiser,
    x_tau: torch.Tensor,              # (B, T, N_lat, D_latent)
    tau_index: torch.Tensor,          # (B, T) long
    step_index: torch.Tensor,         # (B, T) long
    discrete_actions: Optional[torch.Tensor] = None,    # (B, T, n_d)
    actions: Optional[torch.Tensor] = None,  # (B, T, n_c)
    no_grad: bool = False,
) -> torch.Tensor:
    """
    Wrapper around DreamerV4Denoiser, returning z_hat ∈ ℝ^{B,T,N_lat,D_latent}.
    """
    if no_grad:
        with torch.no_grad():
            return denoiser(
                latent_tokens=x_tau,
                diffusion_step=tau_index,
                shortcut_length=step_index,
                discrete_actions=discrete_actions,
                actions=actions,
                causal=True,
            )
    else:
        return denoiser(
            latent_tokens=x_tau,
            diffusion_step=tau_index,
            shortcut_length=step_index,
            discrete_actions=discrete_actions,
            actions=actions,
            causal=True,
        )


# ------------------------------------------------------------
# 7. Shortcut / bootstrap loss (Eq. 7 + Eq. 8)
# ------------------------------------------------------------

def compute_bootstrap_diffusion_loss(
    info: dict,
    denoiser: DreamerV4Denoiser,
    discrete_actions: Optional[torch.Tensor] = None,     # (B, T, n_d)
    actions: Optional[torch.Tensor] = None,   # (B, T, n_c)
):
    """
    info: dict from ForwardDiffusionWithShortcut.forward, with keys:
      x                   : (B, T, N_lat, D_latent)
      x_tau               : (B, T, N_lat, D_latent)
      tau                 : (B, T) float
      step                : (B, T) float
      tau_index           : (B, T) long
      step_index          : (B, T) long  (0 ↔ d_min, >0 ↔ larger steps)
      half_step_index     : (B, T) long
      tau_plus_half_index : (B, T) long
      d_min               : scalar float

    Returns:
      flow_loss      : scalar (flow branch, d = d_min)
      bootstrap_loss : scalar (bootstrap branch, d > d_min)
    """
    x = info["x"]                         # (B, T, N_lat, D_lat)
    x_tau = info["x_tau"]                 # (B, T, N_lat, D_lat)
    tau = info["tau"]                     # (B, T) float
    step = info["step"]                   # (B, T) float
    tau_index = info["tau_index"]         # (B, T) long
    step_index = info["step_index"]       # (B, T) long (0..max_pow2)
    half_step_index = info["half_step_index"]              # (B, T) long
    tau_plus_half_index = info["tau_plus_half_index"]      # (B, T) long
    d_min = info["d_min"]                 # scalar float

    B, T, N_lat, D_lat = x.shape

    tau_b = tau.view(B, T, 1, 1)          # (B, T, 1, 1)
    step_b = step.view(B, T, 1, 1)        # (B, T, 1, 1)

    # -------------------------------------------------
    # 1) Big-step prediction:  \hat z_1 = f_θ(z_τ, τ, d)
    # -------------------------------------------------
    z_hat = apply_denoiser(
        denoiser,
        x_tau,                 # (B, T, N_lat, D_lat)
        tau_index,             # (B, T)
        step_index,            # (B, T)
        discrete_actions=discrete_actions,
        actions=actions,
        no_grad=False,
    )                           # -> (B, T, N_lat, D_lat)

    # =================================================
    # 2) Flow loss branch (d = d_min): || ẑ_1 - z_1 ||^2
    # =================================================
    flow_residual = z_hat - x                      # (B, T, N_lat, D_lat)
    # average over tokens & dims: (B, T)
    flow_sq = flow_residual.pow(2).mean(dim=(-1, -2))

    # d = d_min ↔ step_index == 0 in this convention
    mask_small = (step_index == 0).float()         # (B, T), 1 where d = d_min

    # ramp weight w(τ) from Eq. (8), τ ∈ [0, 1]
    w_tau = ramp_weight(tau)                       # (B, T)

    flow_loss_per = w_tau * flow_sq * mask_small  # (B, T)
    denom_flow = mask_small.sum().clamp_min(1.0)  # avoid div by 0
    flow_loss = flow_loss_per.sum() / denom_flow  # scalar

    # =================================================
    # 3) Bootstrap branch (d > d_min):
    #    v_target = (b1 + b2)/2, with
    #    b1 = [f(z_τ, τ, d/2) - z_τ] / (1 - τ)
    #    b2 = [f(z', τ + d/2, d/2) - z'] / (1 - (τ + d/2))
    #    Loss: (1 - τ)^2 || v_hat - sg(v_target) ||^2 * w(τ)
    # =================================================
    with torch.no_grad():
        # ------- First half-step -------
        f1 = apply_denoiser(
            denoiser,
            x_tau,                  # (B, T, N_lat, D_lat)
            tau_index,              # (B, T)  use same τ
            half_step_index,        # (B, T)  d/2 index
            discrete_actions=discrete_actions,
            actions=actions,
            no_grad=True,
        )                           # (B, T, N_lat, D_lat)

        # b1: (B, T, N_lat, D_lat)
        b1 = (f1 - x_tau) / (1.0 - tau_b)

        # z' = z_τ + b1 * d/2
        z_prime = x_tau + b1 * (step_b / 2.0)      # (B, T, N_lat, D_lat)

        # ------- Second half-step -------
        f2 = apply_denoiser(
            denoiser,
            z_prime,                # (B, T, N_lat, D_lat)
            tau_plus_half_index,    # (B, T)  τ + d/2
            half_step_index,        # (B, T)  d/2
            discrete_actions=discrete_actions,
            actions=actions,
            no_grad=True,
        )                           # (B, T, N_lat, D_lat)

        denom2 = 1.0 - (tau_b + step_b / 2.0)      # (B, T, 1, 1)
        b2 = (f2 - z_prime) / denom2               # (B, T, N_lat, D_lat)

        # v_target: average of the two half-step velocities
        v_target = 0.5 * (b1 + b2)                 # (B, T, N_lat, D_lat)  (sg via no_grad)

    # predicted velocity from big step:
    v_hat = (z_hat - x_tau) / (1.0 - tau_b)        # (B, T, N_lat, D_lat)

    boot_err = v_hat - v_target                    # (B, T, N_lat, D_lat)

    # x-space scaling (1 - τ)^2 and ramp weight w(τ)
    boot_sq = ((1.0 - tau_b) ** 2 * boot_err.pow(2)).mean(dim=(-1, -2))  # (B, T)

    mask_large = (step_index > 0).float()          # (B, T)  d > d_min
    boot_loss_per = w_tau * boot_sq * mask_large   # (B, T)

    denom_boot = mask_large.sum().clamp_min(1.0)
    bootstrap_loss = boot_loss_per.sum() / denom_boot

    return flow_loss, bootstrap_loss