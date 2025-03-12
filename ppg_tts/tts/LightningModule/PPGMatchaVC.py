import numpy as np
import os
import json
import lightning as L
import torch
import wandb
from ...models import PPGMatcha, PPGMatchaV2
from ...utils import plot_mel, plot_tensor_wandb, WarmupCosineAnnealing

class PPGMatchaVC(L.LightningModule):
    def __init__(self,
                 pitch_stats: str,
                 ppg_dim: int,
                 encode_dim: int,
                 spk_emb_size: int,
                 spk_emb_enc_dim: int,
                 num_encoder_layers: int,
                 num_prenet_layers: int,
                 num_hidden_layers: int,
                 decoder_num_mid_block: int,
                 decoder_num_block: int,
                 pitch_emb_size: int,
                 dropout: float=0.1,
                 target_dim: int=80,
                 no_ctc: bool=False,
                 sigma_min: float=1e-4,
                 transformer_type: str='transformer',
                 hidden_transformer_type: str='conformer',
                 encode_transformer_type: str='roformer',
                 nhead: int=4,
                 hidden_kernel_size: int = 5,
                 pre_kernel_size: int = 3,
                 lr: float=1e-4,
                 gamma: float=0.98,
                 diff_steps: int=10,
                 temperature: float=0.667,
                 lr_scheduler_interval: int = 1500,
                 warmup_steps: int = 50000,
                 **kwargs):
        super().__init__()

        with open(pitch_stats, "r") as reader:
            self.pitch_stats = json.load(reader)

        self.save_hyperparameters()

        self.lr = lr
        self.gamma = gamma
        self.no_ctc = no_ctc
        self.diffusion_steps = diff_steps
        self.temperature = temperature
        self.pitch_min = self.pitch_stats['pitch_min']
        self.pitch_max = self.pitch_stats['pitch_max']
        self.lr_scheduler_interval = lr_scheduler_interval
        self.warmup_steps = warmup_steps

        self.model = PPGMatcha(
            ppg_dim=ppg_dim,
            encode_dim=encode_dim,
            spk_emb_size=spk_emb_size,
            spk_emb_enc_dim=spk_emb_enc_dim,
            num_encoder_layers=num_encoder_layers,
            num_prenet_layers=num_prenet_layers,
            num_hidden_layers=num_hidden_layers,
            decoder_num_block=decoder_num_block,
            decoder_num_mid_block=decoder_num_mid_block,
            pitch_min=self.pitch_min,
            pitch_max=self.pitch_max,
            pitch_emb_size=pitch_emb_size,
            dropout=dropout,
            target_dim=target_dim,
            sigma_min=sigma_min,
            transformer_type=transformer_type,
            hidden_transformer_type=hidden_transformer_type,
            encode_transformer_type=encode_transformer_type,
            nhead=nhead,
            hidden_kernel_size=hidden_kernel_size,
            pre_kernel_size=pre_kernel_size,
        )

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self.model.forward(
            x=batch['ppg'],
            spk_emb=batch['spk_emb'],
            pitch_target=batch['log_F0'],
            v_flag=batch['v_flag'],
            mel_target=batch['mel'],
            mel_mask=batch['mel_mask'],
            x_mask=batch['ppg_mask'],
        )

        self.log_dict({
            "train/diffusion_loss": loss,
        })

        return loss
        
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        
        pred_mel = self.model.synthesis(
            x=batch['ppg'],
            spk_emb=batch['spk_emb'],
            pitch_target=batch['log_F0'],
            v_flag=batch['v_flag'],
            mel_mask=batch['mel_mask'],
            diff_steps=self.diffusion_steps,
            temperature=self.temperature,
            x_mask=batch['ppg_mask'],
        )

        mel_loss = torch.nn.functional.l1_loss(pred_mel, batch['mel'])

        self.log_dict({
            "val/mel_loss": mel_loss,
        })

        if batch_idx == 0:
            for sample_id in range(2):
                mel = pred_mel[sample_id]

                mel = mel[:batch['energy_len'][sample_id], :] # (T, 80)

                # log mel to wandb
                self.logger.experiment.log({
                    f"generate_mel/{sample_id}": wandb.Image(plot_tensor_wandb(mel.transpose(-1, -2).squeeze().cpu()),
                                                             caption=f"generated mel in Epoch {self.current_epoch}")
                })
        
        return mel_loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        
        pred_mel = self.model.synthesis(
            x=batch['ppg'],
            spk_emb=batch['spk_emb'],
            pitch_target=batch['log_F0'],
            v_flag=batch['v_flag'],
            mel_mask=batch['mel_mask'],
            diff_steps=self.diffusion_steps,
            temperature=self.temperature,
            x_mask=batch['ppg_mask'],
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
        pred_mel = self.model.synthesis(
            x=batch['ppg'],
            spk_emb=batch['spk_emb'],
            pitch_target=batch['log_F0'],
            v_flag=batch['v_flag'],
            mel_mask=batch['mel_mask'],
            diff_steps=self.diffusion_steps,
            temperature=self.temperature,
            x_mask=batch['ppg_mask'],
        )

        saved_mel = pred_mel.transpose(1,2).detach().cpu().numpy()

        mel_save_dir = self.logger.save_dir + "/mel"

        if not os.path.exists(mel_save_dir):
            os.makedirs(mel_save_dir)

        np.save(f"{mel_save_dir}/{batch['keys'][0]}.npy", saved_mel)

        return pred_mel
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr
        )
        
        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer=optimizer,
        #     gamma=self.gamma
        # )
        scheduler_func = lambda step: WarmupCosineAnnealing(
            step=step,
            warmup_steps=self.warmup_steps,
            total_steps=self.trainer.max_steps,
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=scheduler_func,
        )

        lr_scheduler_config = {'scheduler': lr_scheduler, 'interval': 'step'}
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler_config
        }
    
    def on_fit_end(self):
        self.trainer.test(ckpt_path='best',
                          datamodule=self.trainer.datamodule)

