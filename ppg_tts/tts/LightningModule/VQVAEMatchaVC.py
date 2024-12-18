import numpy as np
import os
import json
import lightning as L
import torch
import wandb
from typing import List
from ...models import VQVAEMatcha
from ...utils import plot_mel, plot_tensor_wandb

class VQVAEMatchaVC(L.LightningModule):
    def __init__(self,
                 pitch_stats: str,
                 ppg_dim: int,
                 ppg_variance: float,
                 encode_dim: int,
                 pitch_emb_size: int,
                 spk_emb_size: int,
                 spk_emb_enc_dim: int,
                 num_emb: int,
                 decoder_num_block: int=1,
                 decoder_num_mid_block: int=2,
                 dropout: float=0.1,
                 target_dim: int=80,
                 sigma_min: float=1e-4,
                 transformer_type: str='transformer',
                 lr: float=1e-4,
                 gamma: float=0.98,
                 no_ctc: bool=False,
                 diff_steps: int=10,
                 temperature: float=0.667,
                 ae_kernel_sizes: List[int] = [3,3,1],
                 ae_dilations: List[int] = [2,4,8],
                 first_stage_steps: int = 100000,
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
        self.ppg_variance = ppg_variance

        self.first_stage_steps = first_stage_steps

        self.model = VQVAEMatcha(
            ppg_dim=ppg_dim,
            encode_dim=encode_dim,
            spk_emb_size=spk_emb_size,
            spk_emb_enc_dim=spk_emb_enc_dim,
            dropout=dropout,
            target_dim=target_dim,
            num_emb=num_emb,
            no_ctc=no_ctc,
            sigma_min=sigma_min,
            transformer_type=transformer_type,
            decoder_num_block=decoder_num_block,
            decoder_num_mid_block=decoder_num_mid_block,
            pitch_min=self.pitch_min,
            pitch_max=self.pitch_max,
            pitch_emb_size=pitch_emb_size,
            ae_kernel_sizes=ae_kernel_sizes,
            ae_dilations=ae_dilations
        )

        self.first_stage_params = list(self.model.vqvae.parameters()) + \
            list(self.model.spk_enc.parameters())

        self.second_stage_params = list(self.model.cfm.parameters()) + \
            list(self.model.pitch_encoder.parameters()) + \
            list(self.model.rope.parameters()) + \
            list(self.model.channel_mapping.parameters()) + \
            list(self.model.cond_channel_mapping.parameters())

        self.automatic_optimization = False

        self.ae_loss = torch.nn.MSELoss()

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        loss, x_rec, emb_loss, commitment_loss = self.model.forward(
            x=batch['ppg'],
            spk_emb=batch['spk_emb'],
            pitch_target=batch['log_F0'],
            v_flag=batch['v_flag'],
            mel_target=batch['mel'],
            mel_mask=batch['mel_mask']
        )

        ae_loss = self.ae_loss(batch['ppg'], x_rec) / self.ppg_variance

        stage1_opt, stage2_opt = self.optimizers()

        if self.global_step < self.first_stage_steps: # In stage 1, train vqvae
            self.log_dict({
                "train/ae_reconstruct_loss": ae_loss,
                "train/embedding_loss": emb_loss,
                "train/commitment_loss": commitment_loss,
            })
            stage1_opt.optimizer.zero_grad()
            total_loss = ae_loss + emb_loss + commitment_loss
            self.manual_backward(total_loss)
            stage1_opt.step()

            if self.global_step % 1250 == 0:
                sch1, _ = self.lr_schedulers()
                sch1.step()

            return total_loss

        else: # In stage 2, train diffuser
            self.log_dict({
                "train/diffusion_loss": loss,
            })
            stage2_opt.optimizer.zero_grad()
            self.manual_backward(loss=loss)
            stage2_opt.step()

            if self.global_step % 1250 == 0:
                _, sch2 = self.lr_schedulers()
                sch2.step()       
        
            return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        
        pred_mel, x_rec, emb_loss, commitment_loss = self.model.synthesis(
            x=batch['ppg'],
            spk_emb=batch['spk_emb'],
            pitch_target=batch['log_F0'],
            v_flag=batch['v_flag'],
            mel_mask=batch['mel_mask'],
            diff_steps=self.diffusion_steps,
            temperature=self.temperature
        )

        mel_loss = torch.nn.functional.l1_loss(pred_mel, batch['mel'])

        ae_loss = self.ae_loss(batch['ppg'], x_rec) / self.ppg_variance

        self.log_dict({
            "val/mel_loss": mel_loss,
            "val/ae_reconstruct_loss": ae_loss,
            "val/embedding_loss": emb_loss,
            "val/commitment_loss": commitment_loss,
        })

        if batch_idx == 0:
            for sample_id in range(2):
                mel = pred_mel[sample_id]

                mel = mel[:batch['energy_len'][sample_id], :] # (T, 80)

                # log mel to wandb
                self.logger.experiment.log({
                    f"generate_mel/{sample_id}": wandb.Image(plot_tensor_wandb(mel.squeeze().cpu()),
                                                             caption=f"generated mel in Epoch {self.current_epoch}")
                })
        
        return mel_loss + ae_loss + emb_loss + commitment_loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        
        pred_mel, _, _, _ = self.model.synthesis(
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
        pred_mel, _, _, _ = self.model.synthesis(
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
        stage1_optimizer = torch.optim.AdamW(
            self.first_stage_params,
            lr=self.lr
        )

        stage2_optimizer = torch.optim.AdamW(
            self.second_stage_params,
            lr=self.lr
        )
        
        stage1_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=stage1_optimizer,
            gamma=self.gamma
        )

        stage2_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=stage2_optimizer,
            gamma=self.gamma
        )
        return [stage1_optimizer, stage2_optimizer], [stage1_lr_scheduler, stage2_lr_scheduler]
    
    def on_fit_end(self):
        self.trainer.test(ckpt_path='best',
                          datamodule=self.trainer.datamodule)

