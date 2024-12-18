import torch
from einops import rearrange
from torch import nn
from typing import Tuple, List

class ResidualConvLayer(nn.Module):
    def __init__(self,
                 channels: int,
                 kernel_size: int,
                 dilation: int,
                 cond_channel: int=None,
                 instance_norm: bool=False):
        """
        A residual convolution layer with customizable kernel size and dilation.

        Args:
            channels (int): Number of input and output channels.
            kernel_size (int): Size of the convolution kernel.
            dilation (int): Dilation rate for the convolution.
        """
        super(ResidualConvLayer, self).__init__()

        # Padding to ensure the output length matches the input length
        padding = (kernel_size - 1) // 2 * dilation

        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )
        if instance_norm:
            self.norm = nn.InstanceNorm1d(channels)
        else:
            self.norm = nn.BatchNorm1d(channels)
        self.activation = nn.ReLU()

        if cond_channel is not None:
            self.cond_path = nn.Sequential(
                nn.Conv1d(
                    in_channels=cond_channel,
                    out_channels=channels * 2,
                    kernel_size=1,
                ),
                nn.ReLU(),
            )

    def forward(self, x: torch.Tensor, cond: torch.Tensor=None) -> torch.Tensor:
        """Forward pass for the residual convolution layer."""

        if cond is not None:
            cond = rearrange(cond, 'b s -> b s 1')
            # FiLM layer between cond and x:
            z_cond = self.cond_path(cond)

            # Use FiLM to force using speaker information here
            cond_b, cond_a = torch.chunk(z_cond, 2, dim=1)
            x = cond_a * x + cond_b

        # Apply convolution, normalization, and activation
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)

        # Add the residual connection
        return x + out


class AutoEncoder(nn.Module):
    def __init__(self,
                 input_channel: int,
                 hidden_channel: int,
                 cond_channel: int,
                 kernel_sizes: List[int],
                 dilations: List[int],
                 dropout: float=0.1):
        super().__init__()

        assert len(kernel_sizes) == len(dilations), "The kernel_sizes and dilations are not compatible"

        self.enc = nn.ModuleList()

        self.enc.append(
            nn.Conv1d(
                in_channels=input_channel,
                out_channels=input_channel // 4,
                padding=1,
                kernel_size=3,
            )
        )

        self.enc.append(
            nn.ReLU()
        )

        for ks, d in zip(kernel_sizes, dilations):
            self.enc.append(
                ResidualConvLayer(input_channel // 4, ks, d, instance_norm=True)
            )

        self.enc.append(
            nn.Conv1d(
                in_channels=input_channel // 4,
                out_channels=hidden_channel,
                kernel_size=1,
            )
        )


        self.dec = nn.ModuleList()

        self.dec.append(
            nn.Conv1d(
                in_channels=hidden_channel,
                out_channels=input_channel // 4,
                kernel_size=1,
            )
        )

        self.dec.append(
            nn.ReLU()
        )

        for ks, d in zip(reversed(kernel_sizes), reversed(dilations)):
            self.dec.append(
                ResidualConvLayer(input_channel // 4, ks, d, cond_channel)
            )

        self.output = nn.Conv1d(
            in_channels=input_channel // 4,
            out_channels=input_channel,
            kernel_size=1,
        )

    def forward(self,
                content: torch.Tensor,
                condition: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            content: content representation from the content encoder of shape (B, T, E)
            condition: condition of shape (B, T, E')
            mask: mask of shape (B, T)
        Return:
            z: hidden representation of shape (B, T, Hid)
            x: reconstruct signal with the same shape as content
        """
        z = rearrange(content, 'b t e -> b e t')

        mask = rearrange(mask, 'b t -> b 1 t')

        for layer in self.enc:
            z = layer(z)

        z_dec = z

        for i, layer in enumerate(self.dec):
            if i < 2:
                z_dec = layer(z_dec)
            else:
                z_dec = layer(z_dec, condition)

        z_dec = self.output(z_dec)

        return rearrange(z.masked_fill(mask, 0.0), 'b h t -> b t h'),\
            rearrange(z_dec.masked_fill(mask, 0.0), 'b e t -> b t e')
