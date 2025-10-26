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
        w = (self.ema_alpha ** (L - 1 - idx))  # newer points weight larger if alpha<1 ? 
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
            lam = x.new_ones((1,1,C)) 

        x = (x - lam * center) / (lam * scale + self.eps)

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
            y = (y - b) / (w + 1e-8)

        x = y * (lam * scale + self.eps) + lam * center
        return x
