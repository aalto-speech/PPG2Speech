import json
import lightning as L
import torch
from pathlib import Path
from .models import ConformerTTS
from torch.utils.data.dataloader import DataLoader
from .dataset import PersoDatasetWithConditions, PersoCollateFn

class PersoDataModule(L.LightningDataModule):
    def __init__(self, 
                 data_dir: str="./data",
                 batch_size: int=16):
        super().__init__()
        self.data_dir = data_dir
        self.train_dir = Path(data_dir) / "train"
        self.val_dir = Path(data_dir) / "val"
        self.test_dir = Path(data_dir) / "test"
        self.batch_size = batch_size

    # def prepare_data(self):
    #     raise NotImplementedError("Please use ./scripts/perso_data.sh for data preparation.")
    
    def setup(self, stage: str):
        if stage == 'fit':
            self.train = PersoDatasetWithConditions(self.train_dir)
            self.val = PersoDatasetWithConditions(self.val_dir)
        elif stage == 'test' or stage == 'predict':
            self.test = PersoDatasetWithConditions(self.test_dir)

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          num_workers=8,
                          collate_fn=PersoCollateFn)
    
    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size=self.batch_size,
                          num_workers=8,
                          collate_fn=PersoCollateFn)
    
    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=8,
                          collate_fn=PersoCollateFn)
    
    def predict_dataloader(self):
        return DataLoader(self.test,
                          batch_size=1,
                          collate_fn=PersoCollateFn)

class ConformerTTSModel(L.LightningModule):
    def __init__(self,
                 mel_loss: torch.nn.Module,
                 energy_loss: torch.nn.Module,
                 pitch_loss: torch.nn.Module,
                 ppg_dim: int,
                 encode_dim: int,
                 num_heads: int,
                 num_layers: int,
                 encode_ffn_dim: int,
                 encode_kernel_size: int,
                 adapter_filter_size: int,
                 adapter_kernel_size: int,
                 n_bins: int,
                 stats_path: str | Path,
                 spk_emb_size: int,
                 emb_hidden_size: int,
                 dropout: float=0.1,
                 target_dim:int=80,
                 backend: str="torchaudio",):
        super().__init__()

        self.save_hyperparameters()

        self.mel_loss = mel_loss
        self.energy_loss = energy_loss
        self.pitch_loss = pitch_loss

        with open(stats_path, "r") as reader:
            stats = json.load(reader)

        self.model = ConformerTTS(
            ppg_dim=ppg_dim,
            encode_dim=encode_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            encode_ffn_dim=encode_ffn_dim,
            encode_kernel_size=encode_kernel_size,
            adapter_filter_size=adapter_filter_size,
            adapter_kernel_size=adapter_kernel_size,
            n_bins=n_bins,
            spk_emb_size=spk_emb_size,
            emb_hidden_size=emb_hidden_size,
            dropout=dropout,
            target_dim=target_dim,
            backend=backend,
            energy_min=stats['energy_min'],
            energy_max=stats['energy_max'],
            pitch_min=stats['pitch_min'],
            pitch_max=stats['pitch_max'],
        )

    def training_step(self, batch, batch_idx):
        # import pdb
        # pdb.set_trace()
        pred_mel = self.model.forward(
            batch["ppg"],
            batch["ppg_len"],
            batch["spk_emb"],
            batch["log_F0"],
            batch["energy"],
            batch["energy_len"],
            batch["mel_mask"]
        )

        l_mel = self.mel_loss(pred_mel, batch["mel"])
        # l_pitch = self.pitch_loss(pred_pitch, batch["log_F0"])
        # l_energy = self.energy_loss(pred_energy, batch["energy"])

        # total = l_mel + l_pitch + l_energy

        self.log_dict({
            "train/mel_loss": l_mel,
        #     "train/pitch_loss": l_pitch,
        #     "train/energy_loss": l_energy,
        #     "train/total_loss": l_mel + l_pitch + l_energy
        })
        
        return l_mel

    def validation_step(self, batch, batch_idx):
        pred_mel = self.model.forward(
            batch["ppg"],
            batch["ppg_len"],
            batch["spk_emb"],
            batch["log_F0"],
            batch["energy"],
            batch["energy_len"],
            batch["mel_mask"]
        )

        l_mel = self.mel_loss(pred_mel, batch["mel"])
        # l_pitch = self.pitch_loss(pred_pitch, batch["log_F0"])
        # l_energy = self.energy_loss(pred_energy, batch["energy"])

        # total = l_mel + l_pitch + l_energy

        self.log_dict({
            "val/mel_loss": l_mel,
        #     "train/pitch_loss": l_pitch,
        #     "train/energy_loss": l_energy,
        #     "train/total_loss": l_mel + l_pitch + l_energy
        })
        
        return l_mel

    def test_step(self, batch, batch_idx):
        pred_mel = self.model.forward(
            batch["ppg"],
            batch["ppg_len"],
            batch["spk_emb"],
            batch["log_F0"],
            batch["energy"],
            batch["energy_len"],
            batch["mel_mask"]
        )

        l_mel = self.mel_loss(pred_mel, batch["mel"])
        # l_pitch = self.pitch_loss(pred_pitch, batch["log_F0"])
        # l_energy = self.energy_loss(pred_energy, batch["energy"])

        # total = l_mel + l_pitch + l_energy

        self.log_dict({
            "test/mel_loss": l_mel,
        #     "train/pitch_loss": l_pitch,
        #     "train/energy_loss": l_energy,
        #     "train/total_loss": l_mel + l_pitch + l_energy
        })
        
        return l_mel

    def predict_step(self, batch, batch_idx):
        raise NotImplementedError("Not implementation for prediction yet. Need Vocoder.")
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=4)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            patience=2,
            factor=0.5
        )
        return {"optimizer": optimizer,
                "lr_scheduler": lr_scheduler,
                "monitor": "val/total_loss"}
