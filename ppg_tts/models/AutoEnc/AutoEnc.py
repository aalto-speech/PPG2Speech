import torch
from torch import nn
from typing import Tuple

class AutoEncoder(nn.Module):
    def __init__(self,
                 input_channel: int,
                 hidden_channel: int,
                 cond_channel: int):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channel,
                out_channels=input_channel // 4,
                padding=1,
                kernel_size=3,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=input_channel // 4,
                out_channels=hidden_channel,
                kernel_size=1,
            )
        )

        self.cond_path = nn.Sequential(
            nn.Conv1d(
                in_channels=cond_channel,
                out_channels=2 * hidden_channel,
                kernel_size=1,
            ),
            nn.ReLU(),
        )

        self.dec = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_channel,
                out_channels=input_channel // 4,
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=input_channel // 4,
                out_channels=input_channel,
                padding=1,
                kernel_size=3,
            )
        )

    def forward(self,
                content: torch.Tensor,
                condition: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            content: content representation from the content encoder of shape (B, E, T)
            condition: condition of shape (B, E', T)
            mask: mask of shape (B, 1, T)
        Return:
            z: hidden representation of shape (B, Hid, T)
            x: reconstruct signal with the same shape as content
        """

        z = self.enc(content)

        z_cond = self.cond_path(condition)

        # Use FiLM to force using speaker information here
        cond_b, cond_a = torch.chunk(z_cond, 2, dim=1)
        z_dec = cond_a * z + cond_b

        x = self.dec(z_dec)

        return z.masked_fill_(mask, 0.0), x.masked_fill_(mask, 0.0)
