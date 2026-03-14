import torch
import torch.nn as nn
from layers.RevIN import RevIN, RevINPlus,RAIN  # 引入实例归一化层 RevIN
from layers.Autoformer_EncDec import series_decomp
from layers.Autoformer_EncDec import series_decomp_learnable
import torch.nn.functional as F         # for softplus, cos, etc.
from scipy.signal import hilbert      # returns numpy arrays
import math

# Hilbert transform: convert real signal to analytic signal
def hilbert_transform(x: torch.Tensor) -> torch.Tensor:
    x_np = x.detach().cpu().numpy()
    z_np = hilbert(x_np, axis=1)
    z = torch.from_numpy(z_np).to(x.device)
    return z

# Phase unwrapping to ensure temporal continuity
def phase_unwrap(phase):
    d = torch.diff(phase, dim=1)
    d_mod = (d + math.pi) % (2 * math.pi) - math.pi
    mask = (d_mod == -math.pi) & (d > 0)
    d_mod = torch.where(mask, torch.full_like(d_mod, math.pi), d_mod)
    phi_adj = d_mod - d
    correction = torch.cumsum(phi_adj, dim=1)
    phase0 = phase[:, :1, :]
    unwrapped = torch.cat([phase0, phase[:, 1:, :] + correction], dim=1)
    return unwrapped

# Robustness evaluation helpers
def wrap_to_pi(phase: torch.Tensor) -> torch.Tensor:
    """Wrap phase into [-pi, pi)."""
    return (phase + math.pi) % (2.0 * math.pi) - math.pi


def add_signal_noise_by_snr(x: torch.Tensor, snr_db: float) -> torch.Tensor:
    signal_power = x.pow(2).mean(dim=1, keepdim=True)  # [B,1,C]
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = torch.randn_like(x) * torch.sqrt(noise_power + 1e-12)
    return x + noise


def build_fast_phase_term(
    L: int,
    device,
    dtype,
    chirp_strength: float = 4.0,
    jump_strength: float = 0.5,
    jump_pos: float = 0.6
) -> torch.Tensor:

    t = torch.linspace(0, 1, L, device=device, dtype=dtype).view(1, L, 1)
    chirp_term = chirp_strength * math.pi * (t ** 2)
    jump_term = jump_strength * math.pi * (t >= jump_pos).to(dtype)
    return chirp_term + jump_term


@torch.no_grad()
def eval_phase_unwrap_robustness(
    seasonal_init: torch.Tensor,
    noise_mode: str = 'none',   # 'none', 'medium', 'strong'
    phase_mode: str = 'stable', # 'stable', 'fast'
    medium_snr_db: float = 20.0,
    strong_snr_db: float = 10.0,
    chirp_strength: float = 4.0,
    jump_strength: float = 0.5,
):

    # -----------------------------------
    # 1) （baseline / target）
    # -----------------------------------
    analytic_ref = hilbert_transform(seasonal_init)
    phase_wrapped_ref = torch.angle(analytic_ref)              # [-pi, pi)
    phase_target = phase_unwrap(phase_wrapped_ref)            # 连续参考相位

    # -----------------------------------
    # 2) noise： seasonal_init
    # -----------------------------------
    seasonal_eval = seasonal_init.clone()

    if noise_mode == 'medium':
        seasonal_eval = add_signal_noise_by_snr(seasonal_eval, medium_snr_db)
    elif noise_mode == 'strong':
        seasonal_eval = add_signal_noise_by_snr(seasonal_eval, strong_snr_db)
    elif noise_mode == 'none':
        pass
    else:
        raise ValueError(f"Unsupported noise_mode: {noise_mode}")

    analytic_eval = hilbert_transform(seasonal_eval)
    phase_wrapped_eval = torch.angle(analytic_eval)

    # -----------------------------------
    # 3) fast_phase
    # -----------------------------------
    if phase_mode == 'fast':
        B, L, C = phase_wrapped_eval.shape
        fast_term = build_fast_phase_term(
            L=L,
            device=phase_wrapped_eval.device,
            dtype=phase_wrapped_eval.dtype,
            chirp_strength=chirp_strength,
            jump_strength=jump_strength
        )  # [1,L,1]

        # add fast term
        phase_target = phase_target + fast_term
        # unwrap  [-pi, pi)
        phase_wrapped_eval = wrap_to_pi(phase_wrapped_eval + fast_term)

    elif phase_mode == 'stable':
        pass
    else:
        raise ValueError(f"Unsupported phase_mode: {phase_mode}")

    phase_pred = phase_unwrap(phase_wrapped_eval)
    err = phase_pred - phase_target
    phase_mae = err.abs().mean().item()
    phase_rmse = torch.sqrt((err ** 2).mean()).item()
    return phase_pred, phase_target, phase_mae, phase_rmse


# Symmetric reflection padding for 1D sequence
def _same_pad_1d(x, k):
    pad_left = k // 2
    pad_right = k - 1 - pad_left
    return F.pad(x, (pad_left, pad_right), mode='reflect')

# Zero-mean phase corrector using depthwise convolution
class ZeroMeanPhaseCorrector(nn.Module):
    """Zero-mean phase correction using depthwise 1D conv"""
    def __init__(self, C: int, k: int = 5, strength_init: float = 0.2):
        super().__init__()
        assert k % 2 == 1
        self.k = k
        self.weight = nn.Parameter(torch.zeros(C, 1, k))
        self.strength = nn.Parameter(torch.tensor(strength_init))

    def forward(self, phase_unwrapped):
        B, L, C = phase_unwrapped.shape
        w = self.weight - self.weight.mean(dim=-1, keepdim=True)
        x = phase_unwrapped.permute(0, 2, 1)
        x_p = _same_pad_1d(x, self.k)
        delta = F.conv1d(x_p, w, stride=1, padding=0, groups=C)
        delta = delta.permute(0, 2, 1)
        gate = torch.tanh(self.strength)
        return phase_unwrapped + gate * delta


# Torch implementation of phase unwrapping
def phase_unwrap_torch(phase: torch.Tensor, time_dim: int = 1, discont: float = math.pi) -> torch.Tensor:
    d = torch.diff(phase, dim=time_dim)
    two_pi = 2.0 * math.pi
    delta_mod = (d + math.pi) % two_pi - math.pi
    mask = (delta_mod == -math.pi) & (d > 0)
    delta_mod = torch.where(mask, torch.full_like(delta_mod, math.pi), delta_mod)
    correction = delta_mod - d
    correction = torch.where(torch.abs(d) >= discont, correction, torch.zeros_like(correction))
    pad_shape = list(phase.shape)
    pad_shape[time_dim] = 1
    pad = torch.zeros(pad_shape, dtype=phase.dtype, device=phase.device)
    unwrapped = phase + torch.cat([pad, torch.cumsum(correction, dim=time_dim)], dim=time_dim)
    return unwrapped

# Trend–Seasonality Coupled module
class TSCoupler(nn.Module):
    r"""
    Strict AM–FM coupling:
      I(t) = α_c · f_β(T(t))_LP · cos(unwrap(φ_season(t)) ⊕ δφ(t) + φ0_c)
    """

    def __init__(self,
                 dropout: float = 0.,
                 beta_init: float = 1.0,
                 C: int = 21,
                 phase_corr_k: int = 15,):
        super().__init__()

        self.C = C
        self.alpha_log = nn.Parameter(torch.zeros(C))
        self.alpha_log02 = nn.Parameter(torch.zeros(C))
        self.phi0 = nn.Parameter(torch.zeros(C))
        self.beta1_log = nn.Parameter(torch.log(torch.tensor(beta_init)))
        self.beta2_log = nn.Parameter(torch.log(torch.tensor(beta_init)))

        self.env_lpf = series_decomp_learnable(kernel_size=25, C=7, mode='gaussian', share_across_channels=False)
        self.phase_corr = ZeroMeanPhaseCorrector(C=C, k=phase_corr_k, strength_init=0.2)

        init_weight_mat = torch.eye(self.C) * 1.0 + torch.randn(self.C, self.C) * 1.0
        self.weight_mat = nn.Parameter(init_weight_mat[None, :, :])

    def f_beta(self, x):
        beta1 = F.softplus(self.beta1_log) + 1e-6
        beta2 = F.softplus(self.beta2_log) + 1e-6
        return beta1 * torch.log1p(torch.exp(beta2 * x))

    def forward(
            self,
            seasonal_init: torch.Tensor,
            trend_init: torch.Tensor,
            noise_mode: str = 'none',  # 'none', 'medium', 'strong'
            phase_mode: str = 'stable',  # 'stable', 'fast'
            medium_snr_db: float = 20.0,
            strong_snr_db: float = 10.0,
            chirp_strength: float = 4.0,
            jump_strength: float = 0.5,
    ):
        B, L, C = seasonal_init.shape
        eps = 1e-6

        # 1) 噪声先加在 seasonal_init 上
        seasonal_eval = seasonal_init
        if noise_mode == 'medium':
            seasonal_eval = add_signal_noise_by_snr(seasonal_eval, medium_snr_db)
        elif noise_mode == 'strong':
            seasonal_eval = add_signal_noise_by_snr(seasonal_eval, strong_snr_db)

        # 2) Hilbert 相位提取
        analytic = hilbert_transform(seasonal_eval)
        phase = torch.angle(analytic)

        # 3) 快速相位变化，在 unwrap 之前加入
        if phase_mode == 'fast':
            fast_term = build_fast_phase_term(
                L=L,
                device=phase.device,
                dtype=phase.dtype,
                chirp_strength=chirp_strength,
                jump_strength=jump_strength
            )
            phase = wrap_to_pi(phase + fast_term)

        # 4) 相位展开
        phase_u = phase_unwrap(phase)

        phi_corr = self.phase_corr(phase_u)
        phi0 = self.phi0.view(1, 1, C)
        phi_corr = phi_corr + phi0

        T_clamped = torch.clamp(trend_init, -10.0, 10.0)
        A_raw = self.f_beta(T_clamped)
        A_lp = A_raw

        alpha = F.softplus(self.alpha_log).view(1, 1, C) + eps
        A_t = alpha * A_lp[0]

        I = A_t * torch.cos(phi_corr)
        I = I.permute(0, 2, 1)
        return I

def _ensure_odd(k: int) -> int:
    return k if (k % 2 == 1) else (k + 1)

# decompose
class decomp_Gaussian(nn.Module):
    def __init__(self, k: int = 25, sigma_init: float = 3.0):
        super().__init__()
        self.k = _ensure_odd(k)
        self.log_sigma = None
        self._built_C = None

    def _build_if_needed(self, C: int, device, dtype, sigma_init: float = 3.):
        if (self.log_sigma is None) or (self._built_C != C):
            self.log_sigma = nn.Parameter(
                torch.full((C,), math.log(sigma_init), device=device, dtype=dtype)
            )
            self._built_C = C

    def _gaussian_kernel(self, C: int, device, dtype):
        half = self.k // 2
        idx = torch.arange(-half, half + 1, device=device, dtype=dtype)
        sigma = torch.exp(self.log_sigma).view(C, 1, 1) + 1e-6
        g = torch.exp(-(idx.view(1, 1, self.k) ** 2) / (2 * sigma * sigma))
        g = g / (g.sum(dim=-1, keepdim=True) + 1e-12)
        return g  # [C,1,k]

    def forward(self, x):  # x: [B,T,C]
        B, T, C = x.shape
        x_ch = x.permute(0, 2, 1)
        self._build_if_needed(C, x.device, x.dtype, sigma_init=3.0)
        w = self._gaussian_kernel(C, x.device, x.dtype)

        pad_left = self.k // 2
        pad_right = self.k - 1 - pad_left
        x_pad = F.pad(x_ch, (pad_left, pad_right), mode='reflect')

        trend = F.conv1d(x_pad, w, stride=1, padding=0, groups=C)
        seasonal = x_ch - trend
        return seasonal.permute(0, 2, 1), trend.permute(0, 2, 1)


def _ensure_odd(k: int) -> int:
    return k if (k % 2 == 1) else (k + 1)

class decomp_Gaussian01(nn.Module):
    def __init__(self, k: int = 25, sigma_init: float = 3.0):
        super().__init__()
        self.k = _ensure_odd(k)
        self.log_sigma = None
        self._built_C = None
        self.sigma_init = sigma_init

    def _build_if_needed(self, C: int, device, dtype):

        if (self.log_sigma is None) or (self._built_C != C):
            self.log_sigma = nn.Parameter(
                torch.full((C,), math.log(self.sigma_init), device=device, dtype=dtype)
            )
            self._built_C = C

    def _gaussian_kernel(self, C: int, device, dtype, k_eff: int):
        half = k_eff // 2
        idx = torch.arange(-half, half + 1, device=device, dtype=dtype)  # [k_eff]
        sigma = torch.exp(self.log_sigma).view(C, 1, 1) + 1e-6           # [C,1,1]
        g = torch.exp(-(idx.view(1, 1, k_eff) ** 2) / (2 * sigma * sigma))  # [C,1,k_eff]
        g = g / (g.sum(dim=-1, keepdim=True) + 1e-12)
        return g  # [C,1,k_eff]

    def forward(self, x: torch.Tensor):

        B, C, T = x.shape
        device, dtype = x.device, x.dtype
        self._build_if_needed(C, device, dtype)
        k_eff = min(self.k, 2 * T - 1)
        if k_eff % 2 == 0:
            k_eff -= 1
        half = k_eff // 2

        w = self._gaussian_kernel(C, device, dtype, k_eff=k_eff)
        x_pad = F.pad(x, (half, half), mode='reflect')
        trend = F.conv1d(x_pad, w, stride=1, padding=0, groups=C)
        seasonal = x - trend
        return seasonal, trend

# DCM module
class DCM(nn.Module):
    def __init__(self, configs):
        super(DCM, self).__init__()
        self.enc_in = configs.enc_in
        kernel_size = configs.moving_avg
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.seq_len = configs.seq_len

        # self.decomp = series_decomp(kernel_size)
        self.decomp_GS = decomp_Gaussian(k=kernel_size)
        self.TS_Coupled = TSCoupler(C=self.enc_in)
        self.I_linear = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.seq_len, self.d_model), nn.GELU(), )
        self.x_linear = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.seq_len, self.d_model), nn.GELU(), )
        # self.I_linear = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.seq_len, self.d_model), nn.LeakyReLU(), )
        # self.x_linear = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.seq_len, self.d_model), nn.LeakyReLU(), )


    def forward(self, x_input):

        x_out = self.x_linear(x_input)
        # seasonal_init, trend_init = self.decomp_LD(x_input.permute(0, 2, 1))
        seasonal_init, trend_init = self.decomp_GS(x_input.permute(0, 2, 1))
        I = self.TS_Coupled(seasonal_init, trend_init)
        I_oupled = self.I_linear(I)
        return x_out,I_oupled

# FEAM module
class FEAM(nn.Module):
    def __init__(self, configs):
        super(FEAM, self).__init__()
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.seq_len = configs.seq_len

        self.channelAggregator = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=4, batch_first=True,dropout=0.5)  # 通道聚合模块
        self.linear = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.d_model, self.d_model), nn.GELU(), )
        # self.linear = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.d_model, self.d_model), nn.LeakyReLU(), )


    def forward(self, x_input, I_oupled):

        I_fused, _ = self.channelAggregator(query=I_oupled , key=x_input, value=x_input)  # 通道聚合操作
        I_fused = self.linear(I_fused + x_input)
        return I_fused

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        self.use_revin = configs.use_revin
        self.revin_layer = RevIN(self.enc_in, affine=True)

        self.DCM = DCM(configs)
        self.FEAM = FEAM(configs)
        self.output_proj = nn.Sequential(nn.Dropout(self.dropout),nn.Linear(self.d_model, self.pred_len),)

    def forward(self, x, x_mark):
        # x = self.revin_layer(x, mode='norm')
        # instance norm

        seq_mean = torch.mean(x, dim=1, keepdim=True)
        seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        x = (x - seq_mean) / torch.sqrt(seq_var)

        x_input = x.permute(0, 2, 1)

        # DCM
        x_out, I_oupled = self.DCM(x_input)
        # FEAM
        FEAM_out = self.FEAM(x_out, I_oupled)

        output = self.output_proj(FEAM_out).permute(0, 2, 1)

        output = output * torch.sqrt(seq_var) + seq_mean
        # output = self.revin_layer(output, mode='denorm')
        return output

