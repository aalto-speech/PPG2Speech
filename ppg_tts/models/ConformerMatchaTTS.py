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
                 decoder_num_mid_block: int,
                 decoder_num_block: int,
                 dropout: float=0.1,
                 target_dim: int=80,
                 no_ctc: bool=False,
                 sigma_min: float=1e-4,
                 transformer_type: str='conformer'):
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

        self.channel_mapping = nn.Sequential(
            nn.Linear(encode_dim, target_dim),
            nn.LayerNorm(target_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.cfm = CFM(
            in_channels=target_dim,
            out_channel=target_dim,
            n_spks=50,
            cfm_params={
                'sigma_min': sigma_min
            },
            spk_emb_dim=spk_emb_size,
            decoder_params={
                'dropout': dropout,
                'down_block_type': transformer_type,
                'mid_block_type': transformer_type,
                'up_block_type': transformer_type,
                'n_blocks': decoder_num_block,
                'num_mid_blocks': decoder_num_mid_block,
            },
        )

    def forward(self,
                x: torch.Tensor,
                spk_emb: torch.Tensor,
                pitch_target: torch.Tensor,
                v_flag: torch.Tensor,
                energy_length: torch.Tensor,
                mel_target: torch.Tensor,
                mel_mask: torch.Tensor):
        """
        Arguments:
            x: input PPG, shape (B, T_ppg, E)
            spk_emb: speaker_embedding, shape (B, E_spk)
            pitch_target: shape (B, T_mel)
            v_flag: shape (B, T_mel)
            energy_length: shape (B,)
            mel_target: shape (B, T, E),
            mel_mask: shape (B, T), bool tensor
        Returns:
            loss
            refined_mel: shape (B, T, 80)
        """

        x = torch.cat([x,
                       pitch_target.unsqueeze(-1),
                       v_flag.unsqueeze(-1)
                       ],
                      dim=-1)
        
        x = self.pre_net(x)

        x_pos_enc = self.rope(x.unsqueeze(1)).squeeze(1)

        x_enc, _ = self.encoder(x_pos_enc, energy_length)

        mu = self.channel_mapping(x_enc)

        if mu.size(1) % 2 == 1:
            mu, mel_mask, mel_target = self._pad_to_even(mu,
                                                         mel_mask,
                                                         mel_target)

        loss, y = self.cfm.compute_loss(
            x1=mel_target.transpose(-1, -2),
            mu=mu.transpose(-1, -2),
            mask=~mel_mask.unsqueeze(1),
            spks=spk_emb
        )

        return loss, y
    
    def _pad_to_even(self,
                     mu: torch.Tensor,
                     mel_mask: torch.Tensor,
                     mel_target: torch.Tensor=None):
        pad_mu = nn.functional.pad(mu,
                                   (0,0,0,1),
                                   mode='constant',
                                   value=0.0)
        pad_mask = nn.functional.pad(mel_mask,
                                     (0,1),
                                     mode='constant',
                                     value=True)

        if mel_target is not None:
            pad_mel_target = nn.functional.pad(mel_target,
                                               (0,0,0,1),
                                               mode='constant',
                                               value=0.0)
        else:
            pad_mel_target = None
            
        return pad_mu, pad_mask, pad_mel_target
    
    def synthesis(self,
                  x: torch.Tensor,
                  spk_emb: torch.Tensor,
                  pitch_target: torch.Tensor,
                  v_flag: torch.Tensor,
                  energy_length: torch.Tensor,
                  mel_mask: torch.Tensor,
                  diff_steps: int=300,
                  temperature: float=0.667):
        """
        Arguments:
            x: input PPG, shape (B, T_ppg, E)
            spk_emb: speaker_embedding, shape (B, E_spk)
            pitch_target: shape (B, T_mel)
            v_flag: shape (B, T_mel)
            energy_length: shape (B,)
            mel_target: shape (B, T, E),
            mel_mask: shape (B, T), bool tensor
        Returns:
            pred_mel: shape (B, T, 80)
        """

        pad_to_odd = False

        x = torch.cat([x,
                       pitch_target.unsqueeze(-1),
                       v_flag.unsqueeze(-1)
                       ],
                      dim=-1)
        
        x = self.pre_net(x)

        x_pos_enc = self.rope(x.unsqueeze(1)).squeeze(1)

        x_enc, _ = self.encoder(x_pos_enc, energy_length)

        mu = self.channel_mapping(x_enc)

        if mu.size(1) % 2 == 1:
            pad_to_odd = True
            mu, mel_mask, _ = self._pad_to_even(mu,
                                                mel_mask)

        pred_mel = self.cfm.forward(
            mu=mu.transpose(-1, -2),
            mask=~mel_mask.unsqueeze(1),
            n_timesteps=diff_steps,
            spks=spk_emb,
            temperature=temperature
        )

        if pad_to_odd:
            pred_mel = pred_mel[:, :, :-1]

        return pred_mel.transpose(-1, -2)
