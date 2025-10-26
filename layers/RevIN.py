# code from https://github.com/ts-kim/RevIN, with minor modifications

import torch
import torch.nn as nn

# from utils.tools import forward_fill


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.mask = None
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str, mask=None):
        # x [b,l,n]
        if mode == 'norm':
            self._get_statistics(x, mask)
            x = self._normalize(x, mask)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x, mask=None):
        self.mask = mask
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            if mask is None:
                self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            else:
                assert isinstance(mask, torch.Tensor)
                # print(type(mask))
                x = x.masked_fill(mask, 0)  # in case other values are filled
                self.mean = (torch.sum(x, dim=1) / torch.sum(~mask, dim=1)).unsqueeze(1).detach()
                # self.mean could be nan or inf
                self.mean = torch.nan_to_num(self.mean, nan=0.0, posinf=0.0, neginf=0.0)

        if mask is None:
            self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
        else:
            self.stdev = (torch.sqrt(torch.sum((x - self.mean) ** 2, dim=1) / torch.sum(~mask, dim=1) + self.eps)
                          .unsqueeze(1).detach())
            self.stdev = torch.nan_to_num(self.stdev, nan=0.0, posinf=None, neginf=None)

    def _normalize(self, x, mask=None):
        self.mask = mask
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean

        x = x / self.stdev

        # x should be zero, if the values are masked
        if mask is not None:
            # forward fill
            # x, mask2 = forward_fill(x, mask)
            # x = x.masked_fill(mask2, 0)

            # mean imputation
            x = x.masked_fill(mask, 0)

        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class RevINPlus(nn.Module):
    """
    Reversible Instance Normalization (improved):
      y = ((x - λ·center) / (λ·scale + eps)) * γ + β
      x = ((y - β) / (γ + tiny)) * (λ·scale + eps) + λ·center
    - λ ∈ (0,1): per-channel learnable gate (partial normalization), via sigmoid
    - center_mode: 'mean' | 'last' | 'ema'
    - scale_mode : 'std'  (hook left for 'mad'/'iqr' if needed)
    - mask-safe statistics on time dimension
    """
    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 affine: bool = True,
                 center_mode: str = 'mean',   # 'mean' | 'last' | 'ema'
                 scale_mode: str = 'std',     # 'std'  (extendable)
                 ema_alpha: float = 0.9,      # for 'ema' center
                 learnable_lambda: bool = True,
                 lambda_init: float = 1.0):
        super().__init__()
        assert center_mode in ('mean', 'last', 'ema')
        assert scale_mode  in ('std',)
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.affine = bool(affine)
        self.center_mode = center_mode
        self.scale_mode  = scale_mode
        self.ema_alpha   = float(ema_alpha)

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias   = nn.Parameter(torch.zeros(self.num_features))
        else:
            self.register_parameter('affine_weight', None)
            self.register_parameter('affine_bias', None)

        # per-channel gate λ in (0,1), initialized near lambda_init
        if learnable_lambda:
            # map initial λ to logit
            lam0 = torch.clamp(torch.tensor(lambda_init, dtype=torch.float32), 0.0 + 1e-4, 1.0 - 1e-4)
            logit0 = torch.log(lam0 / (1 - lam0))
            self.lambda_logit = nn.Parameter(logit0.repeat(self.num_features))
        else:
            self.register_parameter('lambda_logit', None)

        # placeholders for reversible stats (per batch call)
        self._center = None   # [B,1,C]
        self._scale  = None   # [B,1,C]
        self._mask   = None   # [B,L,C] or None

    @staticmethod
    def _count_valid(mask: torch.Tensor, dim=1, keepdim=True) -> torch.Tensor:
        # mask=True means invalid; valid = ~mask
        valid = (~mask).to(torch.float32)
        cnt = valid.sum(dim=dim, keepdim=keepdim)
        return cnt.clamp_min(1.0), valid

    def _ema_center(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Exponential-weighted average over time (vectorized):
          weights t from old->new: w_t ∝ α^t, normalized on valid positions
        """
        B, L, C = x.shape
        device, dtype = x.device, x.dtype
        # weights in increasing time index (0..L-1)
        idx = torch.arange(L, device=device, dtype=dtype).view(1, L, 1)
        w = (self.ema_alpha ** (L - 1 - idx))  # newer points weight larger if alpha<1 ? (可按需互换指数方向)
        if mask is not None:
            cnt, valid = self._count_valid(mask, dim=1, keepdim=True)  # [B,1,C], [B,L,C]
            w = w * valid  # zero-out invalid
            denom = w.sum(dim=1, keepdim=True).clamp_min(1e-12)
        else:
            denom = w.sum(dim=1, keepdim=True)
        num = (x * w).sum(dim=1, keepdim=True)
        center = (num / denom)
        return center

    def _get_statistics(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Compute per-instance per-channel center & scale on time dim.
        x: [B,L,C], mask: [B,L,C]=True for invalid
        """
        B, L, C = x.shape
        assert C == self.num_features, f"num_features={self.num_features}, but x.shape[2]={C}"
        device, dtype = x.device, x.dtype

        self._mask = mask  # save for possible use

        # center
        if self.center_mode == 'last':
            center = x[:, -1:, :]  # [B,1,C]
        elif self.center_mode == 'ema':
            center = self._ema_center(x, mask)
        else:  # 'mean'
            if mask is None:
                center = x.mean(dim=1, keepdim=True)
            else:
                cnt, valid = self._count_valid(mask, dim=1, keepdim=True)  # [B,1,C]
                center = (x * valid).sum(dim=1, keepdim=True) / cnt

        # scale
        if self.scale_mode == 'std':
            if mask is None:
                var = x.var(dim=1, keepdim=True, unbiased=False)
            else:
                # E[(x-center)^2] on valid positions
                xc = x - center
                cnt, valid = self._count_valid(mask, dim=1, keepdim=True)
                var = ((xc * xc) * valid).sum(dim=1, keepdim=True) / cnt
            scale = torch.sqrt(var + self.eps)
        else:
            raise NotImplementedError

        # detach (not learnable) & keep device/dtype
        self._center = center.detach()
        self._scale  = scale.detach()

    def forward(self, x: torch.Tensor, mode: str, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: [B,L,C], mask(optional): [B,L,C] with True for invalid
        mode: 'norm' or 'denorm'
        """
        if mode == 'norm':
            self._get_statistics(x, mask)
            return self._normalize(x, mask)
        elif mode == 'denorm':
            return self._denormalize(x)
        else:
            raise NotImplementedError(f"mode should be 'norm' or 'denorm', got {mode}")

    def _normalize(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, L, C = x.shape
        center = self._center
        scale  = self._scale

        # per-channel λ in (0,1)
        if self.lambda_logit is not None:
            lam = torch.sigmoid(self.lambda_logit).view(1, 1, C)
        else:
            lam = x.new_ones((1,1,C))  # λ=1 → 完全归一化

        x = (x - lam * center) / (lam * scale + self.eps)

        # mask 区域置零（可换成前向填充/均值填充）
        if mask is not None:
            x = x.masked_fill(mask, 0.0)

        if self.affine:
            w = self.affine_weight.view(1,1,C)
            b = self.affine_bias.view(1,1,C)
            x = x * w + b
        return x

    def _denormalize(self, y: torch.Tensor) -> torch.Tensor:
        B, L, C = y.shape
        center = self._center
        scale  = self._scale

        if self.lambda_logit is not None:
            lam = torch.sigmoid(self.lambda_logit).view(1,1,C)
        else:
            lam = y.new_ones((1,1,C))

        if self.affine:
            w = self.affine_weight.view(1,1,C)
            b = self.affine_bias.view(1,1,C)
            # 正确的反仿射（注意分母是 w + tiny，而不是 eps*eps）
            y = (y - b) / (w + 1e-8)

        x = y * (lam * scale + self.eps) + lam * center
        return x


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def _ensure_odd(k: int) -> int:
    return k if (k % 2 == 1) else (k + 1)

class RAIN(nn.Module):
    """
    Reversible Adaptive Instance-Local Normalization
    y = ((x - λ·μ_t) / (λ·σ_t + eps)) * γ + β
    x = ((y - β) / (γ + tiny)) * (λ·σ_t + eps) + λ·μ_t

    - μ_t, σ_t: 用每通道的“可学习低通核”做局部统计（depthwise 1D conv）
    - 核 = 每通道高斯混合(非负、对称、和为1) → 真低通、零相位、保 DC
    - λ: 每通道可学习门控 ∈ (0,1)，控制归一化强度（部分归一化）
    - 严格可逆：缓存 forward 用到的 μ_t、σ_t、λ 供 denorm 使用
    - mask-safe：支持 [B,L,C] 的布尔 mask（True=无效），局部统计时做加权/归一

    Args:
        num_features: C
        k:          局部核长度（奇数）
        K:          高斯混合分量数
        sigma_inits: 各分量初始 σ（长度 K；不足会重复最后一个）
        eps:        数值稳定项
        affine:     是否使用可学习仿射 γ、β
        learnable_lambda: 是否学习 λ
        lambda_init: λ 初值（0~1）
    """
    def __init__(self,
                 num_features: int,
                 k: int = 25,
                 K: int = 2,
                 sigma_inits=(2.5, 6.0),
                 eps: float = 1e-5,
                 affine: bool = True,
                 learnable_lambda: bool = True,
                 lambda_init: float = 1.0):
        super().__init__()
        self.C = int(num_features)
        self.k = _ensure_odd(k)
        self.K = int(K)
        self.eps = float(eps)
        self.affine = bool(affine)

        # --- 可学习仿射 ---
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.C))
            self.affine_bias   = nn.Parameter(torch.zeros(self.C))
        else:
            self.register_parameter('affine_weight', None)
            self.register_parameter('affine_bias', None)

        # --- 每通道 λ 门控 ---
        if learnable_lambda:
            lam0 = torch.clamp(torch.tensor(lambda_init, dtype=torch.float32), 1e-4, 1 - 1e-4)
            logit0 = torch.log(lam0 / (1 - lam0))
            self.lambda_logit = nn.Parameter(logit0.repeat(self.C))
        else:
            self.register_parameter('lambda_logit', None)

        # --- 每通道高斯混合核参数（懒构建到设备&dtype） ---
        if isinstance(sigma_inits, (list, tuple)):
            base = list(sigma_inits) + [sigma_inits[-1]] * (self.K - len(sigma_inits))
            self._sigma_inits = base[:self.K]
        else:
            self._sigma_inits = [float(sigma_inits)] * self.K

        self.mix_logits = nn.Parameter(torch.zeros(self.C, self.K))     # 每通道的混合权重 logits
        init_sigma = torch.tensor(self._sigma_inits, dtype=torch.float32).log()  # [K]
        self.log_sigma = nn.Parameter(init_sigma.unsqueeze(0).repeat(self.C, 1)) # [C,K]

        # --- 为可逆操作缓存 ---
        self._mu_t  = None  # [B,L,C]
        self._sig_t = None  # [B,L,C]

    # 生成每通道混合高斯核（非负、对称、和为1），返回 [C,1,k]
    def _kernel(self, device, dtype):
        half = self.k // 2
        t = torch.arange(-half, half + 1, device=device, dtype=dtype)  # [k]
        sigma = torch.exp(self.log_sigma) + 1e-6                       # [C,K] >0
        t2 = (t ** 2).view(1, 1, self.k)                               # [1,1,k]
        sigma2 = (sigma ** 2).unsqueeze(-1)                            # [C,K,1]
        g = torch.exp(- t2 / (2.0 * sigma2))                           # [C,K,k]
        g = g / (g.sum(dim=-1, keepdim=True) + 1e-12)                  # 每分量归一
        mix = F.softmax(self.mix_logits, dim=-1).unsqueeze(-1)         # [C,K,1] ≥0, 和为1
        w = (mix * g).sum(dim=1)                                       # [C,k]
        w = 0.5 * (w + torch.flip(w, (-1,)))                           # 对称化
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-12)                  # 再归一
        return w.view(self.C, 1, self.k)                               # [C,1,k]

    # 深度可分卷积（反射填充），x_ch: [B,C,L] → conv(..., groups=C)
    @staticmethod
    def _dwconv_reflect(x_ch, w):
        B, C, L = x_ch.shape
        k = w.shape[-1]
        pad_left = k // 2
        pad_right = k - 1 - pad_left
        x_pad = F.pad(x_ch, (pad_left, pad_right), mode='reflect')
        y = F.conv1d(x_pad, w, stride=1, padding=0, groups=C)
        return y  # [B,C,L]

    # 带 mask 的“加权卷积归一化”： (w*(x*valid)) / (w*valid)
    def _masked_smooth(self, x_ch, w, mask=None):
        # x_ch: [B,C,L], mask: [B,L,C] or None
        if mask is None:
            num = self._dwconv_reflect(x_ch, w)         # [B,C,L]
            return num
        else:
            valid = (~mask).to(x_ch.dtype).permute(0,2,1)  # [B,C,L]
            num = self._dwconv_reflect(x_ch * valid, w)    # [B,C,L]
            den = self._dwconv_reflect(valid, w)           # [B,C,L]
            return num / (den + 1e-12)

    def _compute_local_stats(self, x, mask=None):
        """
        计算 μ_t, σ_t（局部）：[B,L,C]，同 dtype/device
        """
        B, L, C = x.shape
        assert C == self.C
        device, dtype = x.device, x.dtype
        w = self._kernel(device, dtype)                    # [C,1,k]

        x_ch  = x.permute(0,2,1)                           # [B,C,L]
        x2_ch = (x * x).permute(0,2,1)                     # [B,C,L]
        mu_ch  = self._masked_smooth(x_ch,  w, mask)       # [B,C,L]
        ex2_ch = self._masked_smooth(x2_ch, w, mask)       # [B,C,L]
        var_ch = torch.clamp(ex2_ch - mu_ch * mu_ch, min=0.0)
        sig_ch = torch.sqrt(var_ch + self.eps)

        mu  = mu_ch.permute(0,2,1).detach()                # [B,L,C]
        sig = sig_ch.permute(0,2,1).detach()               # [B,L,C]
        return mu, sig

    def forward(self, x: torch.Tensor, mode: str, mask: torch.Tensor = None):
        """
        x: [B,L,C], mask: [B,L,C] (True=无效) or None
        mode: 'norm' or 'denorm'
        """
        if mode == 'norm':
            return self._normalize(x, mask)
        elif mode == 'denorm':
            return self._denormalize(x)
        else:
            raise NotImplementedError

    def _normalize(self, x, mask=None):
        B, L, C = x.shape
        # 1) 局部统计
        mu_t, sig_t = self._compute_local_stats(x, mask)   # [B,L,C]
        self._mu_t  = mu_t
        self._sig_t = sig_t

        # 2) 门控强度 λ
        if self.lambda_logit is not None:
            lam = torch.sigmoid(self.lambda_logit).view(1,1,C)  # [1,1,C]
        else:
            lam = x.new_ones((1,1,C))

        # 3) 可逆“部分归一化”
        y = (x - lam * mu_t) / (lam * sig_t + self.eps)

        # 4) 掩码位置设零（或可改为前向填充/均值填充）
        if mask is not None:
            y = y.masked_fill(mask, 0.0)

        # 5) 仿射
        if self.affine:
            w = self.affine_weight.view(1,1,C)
            b = self.affine_bias.view(1,1,C)
            y = y * w + b
        return y

    def _denormalize(self, y):
        assert self._mu_t is not None and self._sig_t is not None, "Call norm before denorm."
        mu_t  = self._mu_t
        sig_t = self._sig_t
        B, L, C = y.shape

        if self.lambda_logit is not None:
            lam = torch.sigmoid(self.lambda_logit).view(1,1,C)
        else:
            lam = y.new_ones((1,1,C))

        if self.affine:
            w = self.affine_weight.view(1,1,C)
            b = self.affine_bias.view(1,1,C)
            y = (y - b) / (w + 1e-8)

        x = y * (lam * sig_t + self.eps) + lam * mu_t
        return x



