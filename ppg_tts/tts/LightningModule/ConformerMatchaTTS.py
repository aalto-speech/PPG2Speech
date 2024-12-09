import numpy as np
import os
import lightning as L
import torch
import json
from pyannote.audio import Model
from ...models import ConformerMatchaTTS
from torch.optim import lr_scheduler as LRScheduler
from ...utils import plot_mel
from vocoder.hifigan.models import Generator
from vocoder.hifigan.env import AttrDict
from vocoder.hifigan.inference_e2e import load_checkpoint

class ConformerMatchaTTSModel(L.LightningModule):
    def __init__(self,
                 auth_token: str,
                 ppg_dim: int,
                 encode_dim: int,
                 encode_heads: int,
                 encode_layers: int,
                 encode_ffn_dim: int,
                 encode_kernel_size: int,
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
                 vocoder_ckpt: str="vocoder/hifigan/ckpt/g_02500000",
                 ):
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.warm_up_steps = warm_up_steps
        self.model_size = encode_ffn_dim
        self.gamma = gamma
        self.no_ctc = no_ctc
        self.diffusion_steps = diff_steps
        self.temperature = temperature

        self.model = ConformerMatchaTTS(
            ppg_dim=ppg_dim,
            encode_dim=encode_dim,
            encode_heads=encode_heads,
            encode_layers=encode_layers,
            encode_ffn_dim=encode_ffn_dim,
            encode_kernel_size=encode_kernel_size,
            spk_emb_size=spk_emb_size,
            dropout=dropout,
            target_dim=target_dim,
            no_ctc=no_ctc,
            sigma_min=sigma_min,
            transformer_type=transformer_type,
            decoder_num_block=decoder_num_block,
            decoder_num_mid_block=decoder_num_mid_block,
        )

        # Load and Freeze Hifi-gan
        # config_file = os.path.join(os.path.split(vocoder_ckpt)[0], 'config.json')
        # with open(config_file) as f:
        #     data = f.read()

        # json_config = json.loads(data)
        # h = AttrDict(json_config)

        # self.generator = Generator(h)

        # state_dict_g = load_checkpoint(vocoder_ckpt, self.device)
        # self.generator.load_state_dict(state_dict_g['generator'])

        # for param in self.generator.parameters():
        #     param.requires_grad = False

        # # Load and Freeze Speaker Embedding model
        # self.spk_emb_model = Model.from_pretrained("pyannote/embedding", use_auth_token=auth_token, strict=False)

        # for param in self.spk_emb_model.parameters():
        #     param.requires_grad = False

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        loss, _ = self.model.forward(
            x=batch['ppg'],
            spk_emb=batch['spk_emb'],
            pitch_target=batch['log_F0'],
            v_flag=batch['v_flag'],
            energy_length=batch['energy_len'],
            mel_target=batch['mel'],
            mel_mask=batch['mel_mask']
        )
        
        self.log_dict({
            "train/diffusion_loss": loss,
        })
        
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        
        with torch.no_grad():
            pred_mel = self.model.synthesis(
                x=batch['ppg'],
                spk_emb=batch['spk_emb'],
                pitch_target=batch['log_F0'],
                v_flag=batch['v_flag'],
                energy_length=batch['energy_len'],
                mel_mask=batch['mel_mask'],
                diff_steps=self.diffusion_steps,
                temperature=self.temperature
            )

            mel_loss = torch.nn.functional.l1_loss(pred_mel, batch['mel'])

        self.log_dict({
            "val/mel_loss": mel_loss
        })
        
        return mel_loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        with torch.no_grad():
            pred_mel = self.model.synthesis(
                x=batch['ppg'],
                spk_emb=batch['spk_emb'],
                pitch_target=batch['log_F0'],
                v_flag=batch['v_flag'],
                energy_length=batch['energy_len'],
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
        with torch.no_grad():
            pred_mel = self.model.synthesis(
                x=batch['ppg'],
                spk_emb=batch['spk_emb'],
                pitch_target=batch['log_F0'],
                v_flag=batch['v_flag'],
                energy_length=batch['energy_len'],
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

