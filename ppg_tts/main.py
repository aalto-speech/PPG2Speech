import torch
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import seed_everything
from .tts.DataModule import PersoDataModule
from .tts.LightningModule import ConformerTTSModel

def cli_main():
    cli = LightningCLI(ConformerTTSModel,
                       PersoDataModule,
                       save_config_kwargs={"overwrite": True})

if __name__ == "__main__":
    seed_everything(17, workers=True)
    torch.set_float32_matmul_precision('high')
    cli_main()