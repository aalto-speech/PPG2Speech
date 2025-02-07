from typing import Optional
import torch
from torch import nn
from einops import rearrange
from speechbrain.lobes.models.transformer.Transformer import TransformerEncoder
from speechbrain.lobes.models.transformer.Conformer import ConformerEncoder
from speechbrain.nnet.attention import RelPosEncXL
from transformers import RoFormerConfig, RoFormerModel

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
    
class RoFormerWrapper(nn.Module):
    def __init__(self,
                 input_dim: int,
                 ffn_dim: int,
                 nhead: int,
                 dropout: float,
                 nlayers: int,
                 **kwargs,):
        super().__init__()
        assert (input_dim / 2) % nhead == 0, "Wrong input dim"
        self.config = RoFormerConfig(
            hidden_size=input_dim,
            num_attention_heads=nhead,
            num_hidden_layers=nlayers,
            intermediate_size=ffn_dim,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=0.0,
        )

        self.encoder = RoFormerModel(
            config=self.config
        )

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (B, T, E)
            x_mask: input tensor of shape (B, T)
        Return:
            a tensor of shape (B, T, E)
        """
        mask = (~x_mask).to(torch.float32)
        output = self.encoder.forward(
            inputs_embeds=x,
            attention_mask=mask
        )

        return output.last_hidden_state.masked_fill(
            rearrange(x_mask, 'b t -> b t 1'),
            0.0
        )

class RelPosTransformerWrapper(nn.Module):
    def __init__(self,
                 input_dim: int,
                 ffn_dim: int,
                 nhead: int,
                 dropout: float,
                 nlayers: int,
                 kernel_size: Optional[int] = None,
                 transformer_type: str = 'conformer',
                 **kwargs,
                 ):
        super().__init__()
        self.relative_pe = RelPosEncXL(input_dim)

        if transformer_type == 'transformer':
            self.encoder = TransformerEncoder(
                num_layers = nlayers,
                nhead = nhead,
                d_ffn = ffn_dim,
                d_model = input_dim,
                dropout = dropout,
                attention_type = 'RelPosMHAXL',
            )
        elif transformer_type == 'conformer':
            assert kernel_size is not None, "kernel_size must be specified if using conformer"
            self.encoder = ConformerEncoder(
                num_layers = nlayers,
                d_model = input_dim,
                d_ffn = ffn_dim,
                nhead = nhead,
                dropout = dropout,
                kernel_size = kernel_size,
            )
        else:
            raise ValueError(f"transformer_type {transformer_type} is unknown.")

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (B, T, E)
            x_mask: input tensor of shape (B, T)
        Return:
            a tensor of shape (B, T, E)
        """
        pe = self.relative_pe(
            x = x,
        )

        mask = rearrange(x_mask, 'b t -> b t 1')

        out, _ = self.encoder(
            src = x,
            src_key_padding_mask = x_mask,
            pos_embs = pe,
        )

        return out.masked_fill(mask, 0.0)
