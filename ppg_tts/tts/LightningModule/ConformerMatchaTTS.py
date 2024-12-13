import numpy as np
import os
import json
import lightning as L
import torch
from ...models import ConformerMatchaTTS
from torch.optim import lr_scheduler as LRScheduler
from ...utils import plot_mel

class ConformerMatchaTTSModel(L.LightningModule):
    def __init__(self,
                 pitch_stats: str,
                 ppg_dim: int,
                 encode_dim: int,
                 pitch_emb_size: int,
                 spk_emb_size: int,
                 decoder_num_block: int=1,
                 decoder_num_mid_block: int=2,
                 dropout: float=0.1,
                 target_dim: int=80,
                 sigma_min: float=1e-4,
                 transformer_type: str='conformer',
                 lr: float=1e-4,
                 lr_scheduler: str="plateau",
                 warm_up_steps: int=25000,
                 gamma: float=0.98,
                 no_ctc: bool=False,
                 diff_steps: int=300,
                 temperature: float=0.667,
                 ae_loss_scale: float=1.0):
        super().__init__()

        with open(pitch_stats, "r") as reader:
            self.pitch_stats = json.load(reader)

        self.save_hyperparameters()

        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.warm_up_steps = warm_up_steps
        self.gamma = gamma
        self.no_ctc = no_ctc
        self.diffusion_steps = diff_steps
        self.temperature = temperature
        self.pitch_min = self.pitch_stats['pitch_min']
        self.pitch_max = self.pitch_stats['pitch_max']
        self.ae_loss_scale = ae_loss_scale

        self.model = ConformerMatchaTTS(
            ppg_dim=ppg_dim,
            encode_dim=encode_dim,
            spk_emb_size=spk_emb_size,
            dropout=dropout,
            target_dim=target_dim,
            no_ctc=no_ctc,
            sigma_min=sigma_min,
            transformer_type=transformer_type,
            decoder_num_block=decoder_num_block,
            decoder_num_mid_block=decoder_num_mid_block,
            pitch_min=self.pitch_min,
            pitch_max=self.pitch_max,
            pitch_emb_size=pitch_emb_size,
        )

        self.ae_loss = torch.nn.L1Loss()

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        loss, x_rec = self.model.forward(
            x=batch['ppg'],
            spk_emb=batch['spk_emb'],
            pitch_target=batch['log_F0'],
            v_flag=batch['v_flag'],
            mel_target=batch['mel'],
            mel_mask=batch['mel_mask']
        )

        ae_loss = self.ae_loss(batch['ppg'], x_rec)
        
        self.log_dict({
            "train/diffusion_loss": loss,
            "train/ae_reconstruct_loss": ae_loss,
        })
        
        return loss + self.ae_loss_scale * ae_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        
        pred_mel, x_rec = self.model.synthesis(
            x=batch['ppg'],
            spk_emb=batch['spk_emb'],
            pitch_target=batch['log_F0'],
            v_flag=batch['v_flag'],
            mel_mask=batch['mel_mask'],
            diff_steps=self.diffusion_steps,
            temperature=self.temperature
        )

        mel_loss = torch.nn.functional.l1_loss(pred_mel, batch['mel'])

        ae_loss = self.ae_loss(batch['ppg'], x_rec)

        self.log_dict({
            "val/mel_loss": mel_loss,
            "val/ae_reconstruct_loss": ae_loss,
        })
        
        return mel_loss + self.ae_loss_scale * ae_loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        
        pred_mel, _ = self.model.synthesis(
            x=batch['ppg'],
            spk_emb=batch['spk_emb'],
            pitch_target=batch['log_F0'],
            v_flag=batch['v_flag'],
            mel_mask=batch['mel_mask'],
            diff_steps=self.diffusion_steps,
            temperature=self.temperature
        )

        mel_loss = torch.nn.functional.l1_loss(pred_mel, batch['mel'])

        self.log_dict({
            "test/mel_loss": mel_loss
        })
        
        if batch_idx % 70 == 0:
            mel_figures_path = self.logger.save_dir + "/mel_samples"

            saved_mel = pred_mel.transpose(1,2).detach().cpu().numpy()

            plot_mel(saved_mel, path=mel_figures_path, key=batch['keys'][-1])
        
        return mel_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pred_mel, _ = self.model.synthesis(
            x=batch['ppg'],
            spk_emb=batch['spk_emb'],
            pitch_target=batch['log_F0'],
            v_flag=batch['v_flag'],
            mel_mask=batch['mel_mask'],
            diff_steps=self.diffusion_steps,
            temperature=self.temperature
        )

        saved_mel = pred_mel.transpose(1,2).detach().cpu().numpy()

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

