from lightning.pytorch.cli import LightningCLI
from .ConformerTTS_Perso import ConformerTTSModel, PersoDataModule

def cli_main():
    cli = LightningCLI(ConformerTTSModel, PersoDataModule)

if __name__ == "__main__":
    cli_main()