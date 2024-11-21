import torch
import torchaudio
from torch import nn
from torchaudio.models import Conformer
from .matcha.flow_matching import CFM
from .matcha.RoPE import RotaryPositionalEmbeddings

class ConformerMatchaTTS(nn.Module):
    def __init__(self,
                 ppg_dim: int,
                 encode_dim: int,
                 encode_heads: int,
                 encode_layers: int,
                 encode_ffn_dim: int,
                 encode_kernel_size: int,
                 spk_emb_size: int,
                 dropout: float=0.1,
                 target_dim: int=80,
                 no_ctc: bool=False,
                 sigma_min: float=1e-4):
        super(ConformerMatchaTTS, self).__init__()

        self.no_ctc = no_ctc

        if no_ctc:
            assert ppg_dim == 1024, "Wrong input dimension with no_ctc option"

        self.pre_net = nn.Linear(in_features=ppg_dim+2,
                                 out_features=encode_dim,
                                 bias=True)
        
        self.rope = RotaryPositionalEmbeddings(
            d=encode_dim
        )
        
        self.encoder = Conformer(
            input_dim=encode_dim,
            num_heads=encode_heads,
            ffn_dim=encode_ffn_dim,
            num_layers=encode_layers,
            depthwise_conv_kernel_size=encode_kernel_size
        )

        self.cfm = CFM(
            in_channels=encode_dim,
            out_channel=target_dim,
            n_spks=50,
            cfm_params={
                'sigma_min': sigma_min
            },
            spk_emb_dim=512,
        )

    def forward(self,
                x: torch.Tensor,
                x_length: torch.Tensor,
                spk_emb: torch.Tensor,
                pitch_target: torch.Tensor,
                v_flag: torch.Tensor,
                energy_length: torch.Tensor):
        """
        Arguments:
            x: input PPG, shape (B, T_ppg, E)
            spk_emb: speaker_embedding, shape (B, E_spk)
            pitch_target: shape (B, T_mel)
            v_flag: shape (B, T_mel)
            energy_length: shape (B,)
        Returns:
            prediected_mel: shape (B, T, 80)
            refined_mel: shape (B, T, 80)
        """
        pass