import torch
from torch import nn
from einops import rearrange

class ConvReluNorm(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 n_layers: int,
                 p_dropout: float=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.conv_layers.append(nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2))
        self.norm_layers.append(nn.LayerNorm(hidden_channels))
        self.relu_drop = nn.Sequential(nn.ReLU(), nn.Dropout(p_dropout))
        for _ in range(n_layers - 1):
            self.conv_layers.append(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
            )
            self.norm_layers.append(nn.LayerNorm(hidden_channels))
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (B, T, E)
            x_mask: input tensor of shape (B, T)
        Return:
            a tensor of shape (B, T, E)
        """

        mask = ~rearrange(x_mask, 'b t -> b 1 t')

        for i in range(self.n_layers):
            x = rearrange(x, 'b t c -> b c t')
            x = self.conv_layers[i](x * mask)
            x = rearrange(x, 'b c t -> b t c')
            if i == 0:
                x_org = x
            x = self.relu_drop(
                self.norm_layers[i](x)
            )

        x = rearrange(x, 'b t c -> b c t')
        out = x_org + rearrange(self.proj(x), 'b c t -> b t c')

        mask = rearrange(x_mask, 'b t -> b t 1')

        return out.masked_fill(mask, 0.0)

