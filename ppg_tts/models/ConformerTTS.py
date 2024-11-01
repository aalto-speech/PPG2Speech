import torch
import torchaudio
from torch import nn
from torch.nn.functional import interpolate
from torchaudio.models import Conformer
from speechbrain.lobes.models.transformer.Conformer import ConformerEncoder, ConformerDecoder
from .modules import VarianceAdaptor, SpeakerEmbeddingEncoder

class ConformerTTS(nn.Module):
    def __init__(self,
                 ppg_dim: int,
                 encode_dim: int,
                 num_heads: int,
                 num_layers: int,
                 encode_ffn_dim: int,
                 encode_kernel_size: int,
                 adapter_filter_size: int,
                 adapter_kernel_size: int,
                 n_bins: int,
                 energy_min: float,
                 energy_max: float,
                 pitch_min: float,
                 pitch_max: float,
                 spk_emb_size: int,
                 emb_hidden_size: int,
                 dropout: float=0.1,
                 target_dim:int=80,
                 backend: str="torchaudio"):
        super(ConformerTTS, self).__init__()

        self.pre_net = nn.Linear(in_features=ppg_dim,
                                 out_features=encode_dim,
                                 bias=True)
        
        self.pred_net = nn.Linear(in_features=encode_dim,
                                  out_features=target_dim)
        
        self.spk_emb_enc = SpeakerEmbeddingEncoder(input_size=spk_emb_size,
                                                   model_size=emb_hidden_size,
                                                   output_size=encode_dim)
        
        self.variance_adapter = VarianceAdaptor(input_size=encode_dim,
                                                filter_size=adapter_filter_size,
                                                kernel_size=adapter_kernel_size,
                                                n_bins=n_bins,
                                                energy_min=energy_min,
                                                energy_max=energy_max,
                                                pitch_min=pitch_min,
                                                pitch_max=pitch_max,
                                                dropout=dropout)
        
        assert backend in ["torchaudio", "speechbrain"],\
            "Conformer backend only support torchaudio and speechbrain"
        
        if backend == "torchaudio":
            self.conformer_enc = Conformer(input_dim=encode_dim,
                                           num_heads=num_heads,
                                           num_layers=num_layers,
                                           ffn_dim=encode_ffn_dim,
                                           depthwise_conv_kernel_size=encode_kernel_size,
                                           dropout=dropout)
            
            self.conformer_dec = Conformer(input_dim=encode_dim,
                                           num_heads=num_heads,
                                           num_layers=num_layers,
                                           ffn_dim=encode_ffn_dim,
                                           depthwise_conv_kernel_size=encode_kernel_size,
                                           dropout=dropout)
        else:
            raise NotImplementedError("Speechbrain implement is not supported yet.")
        
    def _interpolate(self,
                     x: torch.Tensor,
                     target_length: int) -> torch.Tensor:
        x = x.permute(0, 2, 1).unsqueeze(-1)

        x_interpolated = interpolate(x, size=(target_length, 1), mode='bilinear', align_corners=True)

        x_interpolated = x_interpolated.squeeze(-1).permute(0, 2, 1)

        return x_interpolated
        
    def forward(self,
                x: torch.Tensor,
                x_length: torch.Tensor,
                spk_emb: torch.Tensor,
                pitch_target: torch.Tensor | None,
                energy_target: torch.Tensor | None,
                energy_length: torch.Tensor,
                mel_mask: torch.Tensor):
        """
        Arguments:
            x: input PPG, shape (B, T_ppg, E)
            spk_emb: speaker_embedding, shape (B, E_spk)
            pitch_target: shape (B, T_mel)
            energy_target: shape (B, T_mel)
            mel_mask: shape (B, T_mel)
        Returns:
            prediected_mel: shape (B, T, 80)
            predicted_pitch: shape (B, T_mel)
            predicted_energy: shape (B, T_mel)
        """
        T_mel = mel_mask.size(1)
        x = self._interpolate(x, T_mel)

        encoded_spk_emb = self.spk_emb_enc(spk_emb)

        z = self.pre_net(x)

        z, z_length = self.conformer_enc(z, energy_length)

        z, predicted_pitch, predicted_energy = \
            self.variance_adapter(z, mel_mask, pitch_target, energy_target)

        z = z + encoded_spk_emb.unsqueeze(1)

        z, z_length = self.conformer_dec(z, z_length)

        predicted_mel = self.pred_net(z)

        return predicted_mel, predicted_pitch, predicted_energy
