import torch
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import seed_everything
from .ConformerTTS_Perso import ConformerTTSModel, PersoDataModule

def cli_main():
    cli = LightningCLI(ConformerTTSModel,
                       PersoDataModule,
                       save_config_kwargs={"overwrite": True})
    
    cli.trainer.test(ckpt_path="best",
                     datamodule=cli.datamodule)

if __name__ == "__main__":
    seed_everything(17, workers=True)
    torch.set_float32_matmul_precision('high')
    cli_main()