import torch
import torch.nn as nn
import math
import warnings
import torch.nn.functional as F
warnings.filterwarnings('ignore')


class HeteroLaplace_loss(nn.Module):
    def __init__(self, num_channels: int, alpha_init: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.log_sigma_c = nn.Parameter(torch.zeros(num_channels))
        self.alpha_raw = nn.Parameter(torch.tensor(float(alpha_init)))
        self._ln2 = math.log(2.0)

    @torch.no_grad()
    def calibrate_from_gt(self, gt: torch.Tensor):

        median_t = gt.median(dim=1, keepdim=True).values  # (B,1,C)
        mad = (gt - median_t).abs().median(dim=1, keepdim=True).values  # (B,1,C)
        b_init = (mad / self._ln2).mean(dim=(0, 1)).clamp_min(self.eps)  # (C,)
        self.log_sigma_c.data = torch.log(torch.expm1(b_init))  # softplus(log_sigma) ≈ b_init

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        B, L, C = pred.shape
        e = (pred - gt).abs()

        device, dtype = pred.device, pred.dtype

        l = torch.arange(1, L + 1, device=device, dtype=dtype).view(1, L, 1)
        sigma_c = F.softplus(self.log_sigma_c).to(device=device, dtype=dtype).view(1, 1, C)
        alpha = F.softplus(self.alpha_raw).to(device=device, dtype=dtype)

        b = sigma_c * torch.pow(l, alpha) + self.eps

        loss = e / b + torch.log(b)
        return loss.mean()


class WeightedL1Loss:
    def __init__(self, alpha, loss_mode):
        self.alpha = alpha
        self.loss_mode = loss_mode

        self.weights = nn.Parameter(torch.ones(1, 96, 1, dtype=torch.float32), requires_grad=True)

        if self.loss_mode == 'L1':
            self.loss_fun = nn.L1Loss(reduction='none')
        elif self.loss_mode == 'L2':
            self.loss_fun = nn.MSELoss(reduction='none')
        elif self.loss_mode == 'L1L2':
            self.loss_fun1 = nn.L1Loss(reduction='none')
            self.loss_fun2 = nn.MSELoss(reduction='none')

    def __call__(self, pred, gt):
        L = pred.shape[1]
        time_steps = torch.arange(1, L + 1, device=pred.device, dtype=pred.dtype)

        decay_weights = time_steps.pow(-self.alpha)
        weights = decay_weights.unsqueeze(0).unsqueeze(-1)

        if self.loss_mode in ['L1', 'L2']:
            loss_vec = self.loss_fun(pred, gt)
            weightedLoss = torch.mean(loss_vec * weights)
        return weightedLoss
