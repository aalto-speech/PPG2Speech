import numpy as np
import os
import lightning as L
import torch
from typing import List
from ...models import ConformerWavenetTTS
from torch.optim import lr_scheduler as LRScheduler
from ...utils import plot_mel

class ConformerWavenetTTSModel(L.LightningModule):
    def __init__(self,
                 mel_loss: torch.nn.Module,
                 ppg_dim: int,
                 encode_dim: int,
                 num_heads: int,
                 num_layers: int,
                 encode_ffn_dim: int,
                 encode_kernel_size: int,
                 spk_emb_size: int,
                 emb_hidden_size: int,
                 wavenet_residual_channels: int,
                 wavenet_skip_channels: int,
                 wavenet_kernel_size: int,
                 wavenet_cond_channel: int,
                 wavenet_dilations: List[int]=[1, 2, 4, 8, 16, 32, 64],
                 dropout: float=0.1,
                 target_dim: int=80,
                 backend: str="torchaudio",
                 lr: float=1e-4,
                 lr_scheduler: str="plateau",
                 warm_up_steps: int=25000,
                 gamma: float=0.98,
                 no_ctc: bool=False,
                 rmse: bool=False,
                 causal: bool=True):
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.warm_up_steps = warm_up_steps
        self.model_size = encode_ffn_dim
        self.gamma = gamma
        self.no_ctc = no_ctc
        self.rmse = rmse

        self.mel_loss = mel_loss

        self.model = ConformerWavenetTTS(
            ppg_dim=ppg_dim,
            encode_dim=encode_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            encode_ffn_dim=encode_ffn_dim,
            encode_kernel_size=encode_kernel_size,
            wavenet_residual_channels=wavenet_residual_channels,
            wavenet_skip_channels=wavenet_skip_channels,
            wavenet_kernel_size=wavenet_kernel_size,
            wavenet_cond_channel=wavenet_cond_channel,
            spk_emb_size=spk_emb_size,
            emb_hidden_size=emb_hidden_size,
            wavenet_dilations=wavenet_dilations,
            dropout=dropout,
            target_dim=target_dim,
            backend=backend,
            no_ctc=self.no_ctc,
            causal=causal
        )

    def training_step(self, batch, batch_idx):
        pred_mel = self.model.forward(
            batch["ppg"],
            batch["ppg_len"],
            batch["spk_emb"],
            batch["log_F0"],
            batch["v_flag"],
            batch["energy_len"],
        )
        mel_mask = batch['mel_mask'].unsqueeze(-1)
        pred_mel.masked_fill_(mel_mask, 0.0)

        l_mel = self.mel_loss(pred_mel, batch["mel"].masked_fill_(mel_mask, 0.0))
        
        if self.rmse and isinstance(self.mel_loss, torch.nn.MSELoss):
            l_mel = torch.sqrt(l_mel + 1e-9)

        self.log_dict({
            "train/mel_loss": l_mel,
        })
        
        return l_mel

    def validation_step(self, batch, batch_idx):
        pred_mel = self.model.forward(
            batch["ppg"],
            batch["ppg_len"],
            batch["spk_emb"],
            batch["log_F0"],
            batch["v_flag"],
            batch["energy_len"],
        )

        mel_mask = batch['mel_mask'].unsqueeze(-1)
        pred_mel.masked_fill_(mel_mask, 0.0)

        l_mel = self.mel_loss(pred_mel, batch["mel"].masked_fill_(mel_mask, 0.0))
        
        if self.rmse and isinstance(self.mel_loss, torch.nn.MSELoss):
            l_mel = torch.sqrt(l_mel + 1e-9)

        self.log_dict({
            "val/mel_loss": l_mel,
        })
        
        return l_mel

    def test_step(self, batch, batch_idx):
        pred_mel = self.model.forward(
            batch["ppg"],
            batch["ppg_len"],
            batch["spk_emb"],
            batch["log_F0"],
            batch["v_flag"],
            batch["energy_len"],
        )

        l_mel = self.mel_loss(pred_mel, batch["mel"])
        
        if self.rmse and isinstance(self.mel_loss, torch.nn.MSELoss):
            l_mel = torch.sqrt(l_mel + 1e-9)

        self.log_dict({
            "test/mel_loss": l_mel,
        })

        if batch_idx % 70 == 0:
            mel_figures_path = self.logger.save_dir + "/mel_samples"

            saved_mel = pred_mel.transpose(1,2).detach().cpu().numpy()

            plot_mel(saved_mel, path=mel_figures_path, key=batch['keys'][-1])
        
        return l_mel

    def predict_step(self, batch, batch_idx):
        with torch.no_grad():
            pred_mel = self.model.forward(
                batch["ppg"],
                batch["ppg_len"],
                batch["spk_emb"],
                batch["log_F0"],
                batch["v_flag"],
                batch["energy_len"],
                batch["mel_mask"]
            )

        saved_mel = pred_mel.transpose(1,2).detach().cpu().numpy()

        # mel_save_dir = "/scratch/work/liz32/hifigan-perso-finetuned/ft_data/mel"
        mel_save_dir = self.logger.save_dir + "/mel"

        if not os.path.exists(mel_save_dir):
            os.makedirs(mel_save_dir)

        np.save(f"{mel_save_dir}/{batch['keys'][0]}.npy", saved_mel)

        return pred_mel
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.lr)
        
        if self.lr_scheduler == 'plateau':
            lr_scheduler = LRScheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                patience=2,
                factor=0.5
            )
        elif self.lr_scheduler == 'noam':
            schedule_fn = lambda s: \
                (self.model_size ** -0.5) * \
                    min((s + 1) ** -0.5, \
                        (s + 1) * self.warm_up_steps ** -1.5)
            lr_scheduler = LRScheduler.LambdaLR(optimizer=optimizer,
                                                 lr_lambda=schedule_fn)
        elif self.lr_scheduler == 'exponential':
            lr_scheduler = LRScheduler.ExponentialLR(optimizer=optimizer,
                                                      gamma=self.gamma)
        return {"optimizer": optimizer,
                "lr_scheduler": lr_scheduler,
                "monitor": "val/mel_loss"}
    
    def on_fit_end(self):
        self.trainer.test(ckpt_path='best',
                          datamodule=self.trainer.datamodule)

