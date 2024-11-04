from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import seed_everything
from .ConformerTTS_Perso import ConformerTTSModel, PersoDataModule

def cli_main():
    cli = LightningCLI(ConformerTTSModel,
                       PersoDataModule,
                       save_config_kwargs={"overwrite": True})

if __name__ == "__main__":
    seed_everything(17, workers=True)
    cli_main()