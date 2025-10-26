import torch
import torch.nn as nn
import torch.nn.functional as F


class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


def _same_pad_1d(x, k):
    # 保持长度不变的 padding（左右尽量对称）
    pad_left = k // 2
    pad_right = k - 1 - pad_left
    return F.pad(x, (pad_left, pad_right), mode='reflect')

class LearnableMovingAvg(nn.Module):
    """
    可学习的 1D 滑动平均（低通）:
    mode='gaussian' : 用离散高斯核，学习 σ（连续可微，等价于“窗口长度”）
    mode='free'     : 直接学习核权重，softmax 归一化，保证非负且和为1
    share_across_channels: 是否跨通道共享同一核（True更稳、更省参）
    """
    def __init__(self, kernel_size:int, C:int, mode:str='gaussian', share_across_channels:bool=True):
        super().__init__()
        assert kernel_size >= 3, "kernel_size >= 3 更合理"
        self.K = kernel_size
        self.C = C
        self.mode = mode
        self.share = share_across_channels

        if mode == 'gaussian':
            # 学习 σ（>0），初始化为 K/2
            sigma0 = torch.tensor(float(kernel_size) / 2.0)
            self.log_sigma = nn.Parameter(torch.log(sigma0))  # sigma = softplus(log_sigma)
            # 预定义网格（中心在 (K-1)/2）
            n = torch.arange(self.K).float()
            self.register_buffer('grid', n - (self.K - 1) / 2.0)  # [K]
        elif mode == 'free':
            # 直接学习核权重（跨通道共享或每通道一套）
            Wshape = (1, 1, self.K) if self.share else (C, 1, self.K)
            self.kernel_logits = nn.Parameter(torch.zeros(Wshape))  # softmax 后即为权重
        else:
            raise ValueError("mode must be 'gaussian' or 'free'")

    def _build_kernel(self):
        if self.mode == 'gaussian':
            sigma = F.softplus(self.log_sigma) + 1e-6                   # >0
            g = torch.exp(-0.5 * (self.grid / sigma).pow(2))            # [K]
            g = g / (g.sum() + 1e-8)                                    # 归一化，非负且和为1
            # 共享核：形状 [1,1,K]；按通道使用 groups=C 的 depthwise 卷积
            return g.view(1, 1, self.K)
        else:
            # free 模式：对最后一维做 softmax，保证非负、和为1
            k = F.softmax(self.kernel_logits, dim=-1)                    # [1,1,K] 或 [C,1,K]
            return k

    def forward(self, x):
        """
        x: [B, L, C]  ->  trend: [B, L, C]
        """
        B, L, C = x.shape
        k = self._build_kernel()                                        # [1,1,K] 或 [C,1,K]

        # 准备输入为 [B,C,L]
        x_c = x.permute(0, 2, 1)                                        # [B,C,L]
        # same padding
        x_p = _same_pad_1d(x_c, self.K)                                 # [B,C,L+K-1]

        if self.mode == 'gaussian' or self.share:
            # 跨通道共享核：用 groups=C 的 depthwise 卷积
            k_exp = k.expand(C, 1, self.K).contiguous()                 # [C,1,K]
        else:
            # 每通道一套核（free模式）
            k_exp = k                                                   # [C,1,K]

        trend = F.conv1d(x_p, k_exp, bias=None, stride=1, padding=0, groups=C)  # [B,C,L]
        trend = trend.permute(0, 2, 1)                                   # [B,L,C]
        return trend

class series_decomp_learnable(nn.Module):
    """
    用 LearnableMovingAvg 实现的可学习序列分解：
    x = trend + residual，其中 trend 为可学习低通滤波输出。
    """
    def __init__(self, kernel_size:int, C:int, mode:str='gaussian', share_across_channels:bool=True):
        super().__init__()
        self.moving_avg = LearnableMovingAvg(kernel_size, C, mode, share_across_channels)

    def forward(self, x):
        """
        x: [B, L, C]
        returns: res, trend  (与原 series_decomp 接口保持一致)
        """
        trend = self.moving_avg(x)                # [B,L,C]
        res = x - trend                           # [B,L,C]
        return res, trend


class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        # self.decomp1 = series_decomp(moving_avg)
        # self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        return res, attn


class Encoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        # self.decomp1 = series_decomp(moving_avg)
        # self.decomp2 = series_decomp(moving_avg)
        # self.decomp3 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x, trend1 = self.decomp1(x)
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend


class Decoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, trend
