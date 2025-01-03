import torch
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import seed_everything
from .tts.DataModule import PersoDataModule, BasicDataModule, LibriTTSRDataModule
from .tts.LightningModule import ConformerTTSModel, ConformerWavenetTTSModel

def cli_main():
    seed_everything(17, workers=True)
    torch.set_float32_matmul_precision('high')
    cli = LightningCLI(save_config_kwargs={"overwrite": True})

if __name__ == "__main__":
    cli_main()