import torch
from einops import repeat
from torch import nn
from .matcha.flow_matching import CFM
from .matcha.RoPE import RotaryPositionalEmbeddings
from .modules import PitchEncoder, SpeakerEmbeddingEncoder
from .AutoEnc import VQVAE
from typing import List, Tuple

class VQVAEMatcha(nn.Module):
    def __init__(self,
                 ppg_dim: int,
                 encode_dim: int,
                 spk_emb_size: int,
                 spk_emb_enc_dim: int,
                 decoder_num_mid_block: int,
                 decoder_num_block: int,
                 pitch_min: float,
                 pitch_max: float,
                 pitch_emb_size: int,
                 num_emb: int,
                 dropout: float=0.1,
                 target_dim: int=80,
                 no_ctc: bool=False,
                 sigma_min: float=1e-4,
                 transformer_type: str='conformer',
                 ae_kernel_sizes: List[int] = [3,3,1],
                 ae_dilations: List[int] = [2,4,8]):
        super(VQVAEMatcha, self).__init__()

        self.no_ctc = no_ctc

        if no_ctc:
            assert ppg_dim == 1024, "Wrong input dimension with no_ctc option"

        self.spk_enc = SpeakerEmbeddingEncoder(
            input_size=spk_emb_size,
            output_size=spk_emb_enc_dim
        )

        self.pitch_encoder = PitchEncoder(
            emb_size=pitch_emb_size,
            pitch_min=pitch_min,
            pitch_max=pitch_max,
        )

        self.vqvae = VQVAE(
            input_channel=ppg_dim,
            hidden_channel=encode_dim,
            cond_channel=spk_emb_enc_dim,
            kernel_sizes=ae_kernel_sizes,
            dilations=ae_dilations,
            num_emb=num_emb,
        )
        
        self.rope = RotaryPositionalEmbeddings(
            d=encode_dim
        )

        self.channel_mapping = nn.Sequential(
            nn.Linear(encode_dim, target_dim),
            nn.LayerNorm(target_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.cond_channel_mapping = nn.Sequential(
            nn.Conv1d(
                in_channels=spk_emb_enc_dim + pitch_emb_size + 1,
                out_channels=encode_dim,
                kernel_size=1
            ),
            nn.ReLU()
        )

        self.cfm = CFM(
            in_channels=target_dim,
            out_channel=target_dim,
            n_spks=50,
            cfm_params={
                'sigma_min': sigma_min
            },
            spk_emb_dim=encode_dim,
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
                mel_target: torch.Tensor,
                mel_mask: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Arguments:
            x: input PPG, shape (B, T_ppg, E)
            spk_emb: speaker_embedding, shape (B, E_spk)
            pitch_target: shape (B, T_mel)
            v_flag: shape (B, T_mel)
            mel_target: shape (B, T, E),
            mel_mask: shape (B, T), bool tensor
        Returns:
            diffusion loss
            reconstructed x
            embedding loss
            commitment loss
        """
        _, T = pitch_target.shape
        enc_spk_emb = self.spk_enc(spk_emb).squeeze() # B,E -> B,E'
        enc_pitch = self.pitch_encoder(pitch_target, v_flag, mel_mask) # B,T,P -> B,T,E_p+1

        cond = torch.cat([
            repeat(enc_spk_emb, 'b e -> b t e', t=T),
            enc_pitch,
        ], dim=-1) # B,T,E'+E_p+1

        cond_enc = self.cond_channel_mapping(cond.transpose(-1, -2)).transpose(-1,-2)

        z_q, x_rec, emb_loss, commitment = self.vqvae(
            x=x,
            cond=enc_spk_emb,
            mask=~mel_mask
        )

        z_diff = z_q.detach()

        z_pos_enc = self.rope(z_diff.unsqueeze(1)).squeeze(1)

        mu = self.channel_mapping(z_pos_enc)

        if mu.size(1) % 2 == 1:
            mu, mel_mask, cond_enc, mel_target = self._pad_to_even(
                mu,
                mel_mask,
                cond_enc,
                mel_target
            )

        loss, _ = self.cfm.compute_loss(
            x1=mel_target.transpose(-1, -2),
            mu=mu.transpose(-1, -2),
            mask=~mel_mask.unsqueeze(1),
            spks=cond_enc.transpose(-1, -2),
        )

        return loss, x_rec, emb_loss, commitment
    
    def _pad_to_even(self,
                     mu: torch.Tensor,
                     mel_mask: torch.Tensor,
                     cond: torch.Tensor,
                     mel_target: torch.Tensor=None):
        pad_mu = nn.functional.pad(mu,
                                   (0,0,0,1),
                                   mode='constant',
                                   value=0.0)
        
        pad_cond = nn.functional.pad(cond,
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
            
        return pad_mu, pad_mask, pad_cond, pad_mel_target
    
    @torch.no_grad()
    def synthesis(self,
                  x: torch.Tensor,
                  spk_emb: torch.Tensor,
                  pitch_target: torch.Tensor,
                  v_flag: torch.Tensor,
                  mel_mask: torch.Tensor,
                  diff_steps: int=300,
                  temperature: float=0.667) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Arguments:
            x: input PPG, shape (B, T_ppg, E)
            spk_emb: speaker_embedding, shape (B, E_spk)
            pitch_target: shape (B, T_mel)
            v_flag: shape (B, T_mel)
            mel_target: shape (B, T, E),
            mel_mask: shape (B, T), bool tensor
        Returns:
            pred_mel: shape (B, T, 80)
            reconstructed x: same shape as x
            embedding loss
            commitment loss
        """

        pad_to_odd = False
        _, T = pitch_target.shape
        enc_spk_emb = self.spk_enc(spk_emb).squeeze() # B,E -> B,E'
        enc_pitch = self.pitch_encoder(pitch_target, v_flag, mel_mask) # B,T,P -> B,T,E_p+1

        cond = torch.cat([
            repeat(enc_spk_emb, 'b e -> b t e', t=T),
            enc_pitch,
        ], dim=-1) # B,T,E'+E_p+1

        cond_enc = self.cond_channel_mapping(cond.transpose(-1, -2)).transpose(-1,-2)

        z_q, x_rec, emb_loss, commitment = self.vqvae(
            x=x,
            cond=enc_spk_emb,
            mask=~mel_mask
        )

        z_diff = z_q.detach()

        z_pos_enc = self.rope(z_diff.unsqueeze(1)).squeeze(1)

        mu = self.channel_mapping(z_pos_enc)

        if mu.size(1) % 2 == 1:
            pad_to_odd = True
            mu, mel_mask, cond_enc, _ = self._pad_to_even(
                mu,
                mel_mask,
                cond_enc,
            )

        pred_mel = self.cfm.forward(
            mu=mu.transpose(-1, -2),
            mask=~mel_mask.unsqueeze(1),
            n_timesteps=diff_steps,
            spks=cond_enc.transpose(-1, -2),
            temperature=temperature
        )

        if pad_to_odd:
            pred_mel = pred_mel[:, :, :-1]

        return pred_mel.transpose(-1, -2), x_rec, emb_loss, commitment
