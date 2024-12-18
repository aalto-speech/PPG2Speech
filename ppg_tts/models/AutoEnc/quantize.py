import torch
import einops
from torch import nn
from typing import Tuple

class QuantizeLayer(nn.Module):
    def __init__(self,
                 num_emb: int,
                 emb_dim: int,
                 beta: float=0.25):
        super(QuantizeLayer, self).__init__()

        self.num_embeddings = num_emb
        self.embedding_dim = emb_dim
        self.beta = beta

        self.embedding = nn.Embedding(
            num_embeddings=num_emb,
            embedding_dim=emb_dim,
        )

        self.embedding.weight.data.uniform_(-1.0 / num_emb, 1.0 / num_emb)

        self.embedding_loss = nn.MSELoss()

        self.commitment_loss = nn.MSELoss()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: encoder output. Tensor of shape (B, T, embedding_dim)
            mask: mask of shape (B, T)
        Return:
            quantize vector of shape (B, T, embedding_dim)
            embedding loss
            commitment loss
        """

        mask = einops.rearrange(mask, "b t -> b t 1")

        x = x.masked_fill(mask, 0.0)

        x_sg = x.detach()

        x_flat = einops.rearrange(x, "b t e -> (b t) 1 e")

        centers = self.embedding.weight

        centers = einops.rearrange(centers, "n e -> 1 n e")

        dist_mat = torch.sqrt((x_flat - centers) ** 2).sum(-1) # (b t) n

        center_idx = torch.argmin(dist_mat, dim = -1) # (b t)

        z_q = einops.rearrange(self.embedding(center_idx), "(b t) e -> b t e", b=x.shape[0])

        z_q = z_q.masked_fill(mask, 0.0)

        embedding_loss = torch.mean((x_sg - z_q) ** 2) #self.embedding_loss(z_q, x_sg)

        commitment_loss = self.beta * torch.mean((x - z_q.detach()) ** 2) # self.beta * self.commitment_loss(x, z_q.detach())

        z_q = x + (z_q - x).detach()

        return z_q, embedding_loss, commitment_loss
