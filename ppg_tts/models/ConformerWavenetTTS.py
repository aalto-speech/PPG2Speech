import torch
import torchaudio
from typing import List
from torch import nn
from torchaudio.models import Conformer
from speechbrain.lobes.models.transformer.Conformer import ConformerEncoder
from speechbrain.lobes.models.Tacotron2 import Postnet
from .modules import SpeakerEmbeddingEncoder
from .wavenet import WaveNet

class ConformerWavenetTTS(nn.Module):
    def __init__(self,
                 ppg_dim: int,
                 encode_dim: int,
                 num_heads: int,
                 num_layers: int,
                 encode_ffn_dim: int,
                 encode_kernel_size: int,
                 wavenet_residual_channels: int,
                 wavenet_skip_channels: int,
                 wavenet_kernel_size: int,
                 wavenet_cond_channel: int,
                 spk_emb_size: int,
                 emb_hidden_size: int,
                 wavenet_dilations: List[int]=[1, 2, 4, 8, 16, 32, 64],
                 dropout: float=0.1,
                 target_dim: int=80,
                 backend: str="torchaudio",
                 no_ctc: bool=False,
                 causal: bool=True):
        super(ConformerWavenetTTS, self).__init__()

        self.no_ctc = no_ctc

        if no_ctc:
            assert ppg_dim == 1024, "Wrong input dimension with no_ctc option"

        self.pre_net = nn.Linear(in_features=ppg_dim+2,
                                 out_features=encode_dim,
                                 bias=True)
        
        # self.post_net = Postnet()
        
        self.spk_emb_enc = SpeakerEmbeddingEncoder(input_size=spk_emb_size,
                                                   model_size=emb_hidden_size,
                                                   output_size=encode_dim)
        
        assert backend in ["torchaudio", "speechbrain"],\
            "Conformer backend only support torchaudio and speechbrain"
        
        if backend == "torchaudio":
            self.conformer_enc = Conformer(input_dim=encode_dim,
                                           num_heads=num_heads,
                                           num_layers=num_layers,
                                           ffn_dim=encode_ffn_dim,
                                           depthwise_conv_kernel_size=encode_kernel_size,
                                           dropout=dropout)
            
            # self.conformer_fusion = Conformer(input_dim=encode_dim * 2,
            #                                   num_heads=num_heads * 2,
            #                                   ffn_dim=encode_ffn_dim * 2,
            #                                   num_layers=num_layers,
            #                                   depthwise_conv_kernel_size=encode_kernel_size,
            #                                   dropout=dropout)
        else:
            raise NotImplementedError("Speechbrain implement is not supported yet.")
        
        self.decoder = WaveNet(input_channels=2 * encode_dim,
                               output_channels=target_dim,
                               residual_channels=wavenet_residual_channels,
                               skip_channels=wavenet_skip_channels,
                               kernel_size=wavenet_kernel_size,
                               dilations=wavenet_dilations,
                               causal=causal,
                               cond_channels=wavenet_cond_channel)
        
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
            mel_mask: shape (B, T_mel)
        Returns:
            prediected_mel: shape (B, T, 80)
            predicted_pitch: shape (B, T_mel)
            predicted_energy: shape (B, T_mel)
        """

        x = torch.cat([x,
                       pitch_target.unsqueeze(-1),
                       v_flag.unsqueeze(-1)
                       ],
                      dim=-1)
        
        # encoded_spk_emb = self.spk_emb_enc(spk_emb)
        
        z = self.pre_net(x)
        
        z, z_length = self.conformer_enc(z, energy_length)
        
        # z = torch.cat([
        #     z,
        #     encoded_spk_emb.unsqueeze(1).repeat(1, z.size(1), 1)
        #     ],
        #     dim=-1)
        
        spk_emb = spk_emb.unsqueeze(-1).repeat((1, 1, z.size(1)))
        
        # z, _ = self.conformer_fusion(z, z_length)
        
        predicted_mel = self.decoder(z.transpose(-1, -2), spk_emb)

        predicted_mel = predicted_mel.transpose(-1, -2)
        
        return predicted_mel
