import torch
from einops import rearrange, repeat
from torch import nn
from typing import List, Tuple
from .AutoEnc import ResidualConvLayer
from .quantize import QuantizeLayer

class VQVAE(nn.Module):
    def __init__(self,
                 input_channel: int,
                 hidden_channel: int,
                 cond_channel: int,
                 kernel_sizes: List[int],
                 dilations: List[int],
                 num_emb: int,
                 beta: float=0.25,
                 dropout: float=0.1):
        super(VQVAE, self).__init__()

        assert len(kernel_sizes) == len(dilations), "The kernel_sizes and dilations are not compatible"

        # Define Encoder:
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

        # Define quantize module:
        self.vq = QuantizeLayer(
            num_emb=num_emb,
            emb_dim=hidden_channel,
            beta=beta,
        )

        # Define decoder:
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

    def forward(self, x: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor) \
    -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: input of shape (B, T, E)
            cond: input of shape (B, E'). Speaker Embedding in our case.
            mask: bool tensor of shape of (B, T)
        Return:
            z_q: quantized signal of shape (B, T', E_q)
            x': reconstructed x of shape (B, T, E)
            embedding loss
            commitment loss
        """
        z = rearrange(x, 'b t e -> b e t')

        for layer in self.enc:
            z = layer(z)

        z = rearrange(z, 'b e t -> b t e')

        z_q, e_loss, c_loss = self.vq.forward(z, mask)

        z_dec = rearrange(z_q, 'b t e -> b e t')

        for i, layer in enumerate(self.dec):
            if i < 2:
                z_dec = layer(z_dec)
            else:
                z_dec = layer(z_dec, cond)

        z_dec = rearrange(self.output(z_dec), 'b e t -> b t e')

        return z_q, z_dec, e_loss, c_loss
