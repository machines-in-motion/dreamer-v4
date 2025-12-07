import math
from typing import Optional
import torch


@torch.no_grad()
def shortcut_euler_step_last_frame(
    denoiser,
    z_tau_seq: torch.Tensor,          # (B, T_cur, N_lat, D_latent)
    tau_seq: torch.Tensor,            # (B, T_cur) float in [0,1]
    d: float,
    K_max: int,
    discrete_actions: Optional[torch.Tensor] = None,   # (B, T_cur, n_d)
    continuous_actions: Optional[torch.Tensor] = None, # (B, T_cur, n_c)
) -> tuple[torch.Tensor, float]:
    """
    One Euler shortcut step for the *last* frame in the sequence.

    Inputs:
      z_tau_seq : current noisy latents for frames [0..T_cur-1]
      tau_seq   : signal levels for those frames
      d         : step size in tau-space, e.g. 0.25
      K_max     : finest grid size for tau indices (same as in training)
    Returns:
      z_tau_seq_new : updated noisy latents (only last frame is changed)
      tau_last_new  : new tau for the last frame
    """
    device = z_tau_seq.device
    B, T_cur, N_lat, D_lat = z_tau_seq.shape

    d_min = 1.0 / float(K_max)

    # --- Convert τ and d to discrete indices for this call ---
    # tau_index: floor(tau / d_min + 0.5) clamped into [0, K_max-1]
    tau_index = torch.clamp(
        (tau_seq / d_min).round().long(),
        min=0,
        max=K_max - 1,
    )  # (B, T_cur)

    # step_index such that d = d_min * 2^step_index
    step_index_scalar = int(round(math.log2(d / d_min)))
    step_index = torch.full(
        (B, T_cur),
        step_index_scalar,
        dtype=torch.long,
        device=device,
    )  # (B, T_cur)

    # --- Run denoiser: predicts clean z_1 for each frame ---
    z_hat_seq = denoiser(
        latent_tokens=z_tau_seq,
        diffusion_step=tau_index,
        shortcut_length=step_index,
        discrete_actions=discrete_actions,
        continuous_actions=continuous_actions,
        causal=True,
    )  # (B, T_cur, N_lat, D_lat)

    # --- Extract last frame and do Euler update in τ-space ---
    tau_last = tau_seq[:, -1]           # (B,)
    z_tau_last = z_tau_seq[:, -1]       # (B, N_lat, D_lat)
    z_hat_last = z_hat_seq[:, -1]       # (B, N_lat, D_lat)

    # v_hat = (z_hat_1 - z_tau) / (1 - tau)
    one_minus_tau = 1.0 - tau_last      # (B,)
    v_hat_last = (z_hat_last - z_tau_last) / one_minus_tau.view(B, 1, 1)

    # Euler step: z_{τ+d} = z_τ + d * v_hat
    z_tau_last_new = z_tau_last + d * v_hat_last
    tau_last_new = tau_last + d

    # Clamp tau <= 1.0 for safety
    tau_last_new = tau_last_new.clamp(max=1.0)

    # Put updated last frame back into sequence
    z_tau_seq_new = z_tau_seq.clone()
    z_tau_seq_new[:, -1] = z_tau_last_new

    return z_tau_seq_new, tau_last_new



@torch.no_grad()
def sample_dreamer4_trajectory(
    denoiser,
    horizon: int,
    context_latents: Optional[torch.Tensor] = None,        # (B, T_ctx, N_lat, D_latent)
    discrete_actions: Optional[torch.Tensor] = None,       # (B, horizon, n_d)
    continuous_actions: Optional[torch.Tensor] = None,     # (B, horizon, n_c)
    K: int = 4,
    tau_ctx: float = 0.1,
) -> torch.Tensor:
    """
    Autoregressively sample a trajectory of latent representations using
    the Dreamer-4-style shortcut dynamics.

    Args:
      denoiser          : DreamerV4Denoiser (trained dynamics model)
      horizon           : total number of frames to generate (including context)
      context_latents   : optional clean latents for first T_ctx frames
                          shape (B, T_ctx, N_lat, D_latent)
      discrete_actions  : optional discrete actions (B, horizon, n_d)
      continuous_actions: optional continuous actions (B, horizon, n_c)
      K                 : number of shortcut steps per frame (default 4)
      tau_ctx           : context corruption signal level (default 0.1)

    Returns:
      z_clean_all: (B, horizon, N_lat, D_latent) clean latent trajectory
    """
    device = next(denoiser.parameters()).device
    K_max = denoiser.cfg.K_max
    d_min = 1.0 / float(K_max)

    # --- Determine batch size and latent shape ---
    if context_latents is not None:
        B, T_ctx, N_lat, D_lat = context_latents.shape
    else:
        T_ctx = 0
        B = None
        N_lat = None
        D_lat = None

    if continuous_actions is not None:
        B_act, T_act, _ = continuous_actions.shape
        if B is None:
            B = B_act
        else:
            assert B == B_act, "Batch size mismatch between context and actions"
        assert T_act >= horizon, "Need at least 'horizon' actions in time dimension"

    if discrete_actions is not None:
        B_disc, T_disc, _ = discrete_actions.shape
        if B is None:
            B = B_disc
        else:
            assert B == B_disc, "Batch size mismatch between discrete and other inputs"
        assert T_disc >= horizon, "Need at least 'horizon' discrete actions"

    if B is None:
        raise ValueError("Need either context_latents or actions to infer batch size")

    # If context_latents is None, we need N_lat and D_lat from somewhere.
    if context_latents is None:
        # You can either pass down N_lat and D_lat explicitly, or derive from denoiser config.
        # Here we just assume you want the same N_lat and D_lat as training:
        N_lat = denoiser.cfg.num_latent_tokens
        D_lat = denoiser.cfg.latent_dim

    # --- Prepare outputs ---
    z_clean_all = torch.zeros(B, horizon, N_lat, D_lat, device=device)

    # Fill in initial context clean latents, if provided
    if T_ctx > 0:
        z_clean_all[:, :T_ctx] = context_latents.to(device)

    # Pre-slice actions to the horizon we care about
    disc_act_full = None
    cont_act_full = None
    if discrete_actions is not None:
        disc_act_full = discrete_actions[:, :horizon].to(device)
    if continuous_actions is not None:
        cont_act_full = continuous_actions[:, :horizon].to(device)

    # --- Helper for context corruption ---
    tau_ctx_tensor = torch.full((B,), tau_ctx, device=device)  # (B,)

    def make_context_inputs(t_end: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build (z_tau_ctx, tau_ctx_seq) for frames [0..t_end-1] from their clean latents.
        Returns:
          z_tau_ctx   : (B, t_end, N_lat, D_lat)
          tau_ctx_seq : (B, t_end) all = tau_ctx
        """
        if t_end == 0:
            return None, None

        z_clean_ctx = z_clean_all[:, :t_end]  # (B, t_end, N_lat, D_lat)
        z0_ctx = torch.randn_like(z_clean_ctx)  # (B, t_end, N_lat, D_lat)

        # z_tau = (1 - tau_ctx) * z0_ctx + tau_ctx * z_clean
        z_tau_ctx = (1.0 - tau_ctx) * z0_ctx + tau_ctx * z_clean_ctx
        tau_seq_ctx = tau_ctx_tensor.view(B, 1).expand(B, t_end)  # (B, t_end)

        return z_tau_ctx, tau_seq_ctx

    # --- Sampling loop over time ---
    d = 1.0 / float(K)  # shortcut step size, e.g. 0.25 for K=4

    for t in range(T_ctx, horizon):
        # 1) Build context part [0..t-1] at tau_ctx
        if t > 0:
            z_tau_ctx, tau_seq_ctx = make_context_inputs(t)
        else:
            z_tau_ctx, tau_seq_ctx = None, None

        # 2) Initialize current frame t at pure noise, tau = 0
        z0_t = torch.randn(B, 1, N_lat, D_lat, device=device)  # (B, 1, N_lat, D_lat)
        z_tau_t = z0_t.clone()
        tau_t = torch.zeros(B, device=device)                  # (B,)

        # 3) Actions for frames [0..t]
        disc_act_t = None
        cont_act_t = None
        if disc_act_full is not None:
            disc_act_t = disc_act_full[:, : t + 1]  # (B, t+1, n_d)
        if cont_act_full is not None:
            cont_act_t = cont_act_full[:, : t + 1]  # (B, t+1, n_c)

        # 4) K shortcut steps to denoise frame t from τ=0 → τ=1
        for _ in range(K):
            # Build full sequence z_tau_seq, tau_seq for [0..t]
            if t > 0:
                # context + current
                z_tau_seq = torch.cat([z_tau_ctx, z_tau_t], dim=1)  # (B, t+1, N_lat, D_lat)
                tau_seq = torch.cat(
                    [tau_seq_ctx, tau_t.view(B, 1)], dim=1
                )  # (B, t+1)
            else:
                z_tau_seq = z_tau_t              # (B, 1, N_lat, D_lat)
                tau_seq = tau_t.view(B, 1)       # (B, 1)

            # Do one Euler shortcut update on the *last* frame
            z_tau_seq, tau_last_new = shortcut_euler_step_last_frame(
                denoiser=denoiser,
                z_tau_seq=z_tau_seq,
                tau_seq=tau_seq,
                d=d,
                K_max=K_max,
                discrete_actions=disc_act_t,
                continuous_actions=cont_act_t,
            )

            # Extract updated last frame and tau
            z_tau_t = z_tau_seq[:, -1:].contiguous()  # (B, 1, N_lat, D_lat)
            tau_t = tau_last_new                      # (B,)

        # After K steps, tau_t ≈ 1, and z_tau_t is our clean latent for frame t
        z_clean_all[:, t : t + 1] = z_tau_t

    return z_clean_all


