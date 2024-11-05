import torch
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from .ConformerTTS_Perso import ConformerTTSModel, PersoDataModule

def cli_main():
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    ck_callback = ModelCheckpoint(dirpath="./exp",
                                  monitor="val/mel_loss",
                                  save_top_k=3,
                                  save_on_train_epoch_end=False,
                                  auto_insert_metric_name=True,
                                  mode="min",
                                  save_last=True)
    cli = LightningCLI(ConformerTTSModel,
                       PersoDataModule,
                       save_config_kwargs={"overwrite": True},
                       callbacks=[lr_monitor, ck_callback])

if __name__ == "__main__":
    seed_everything(17, workers=True)
    torch.set_float32_matmul_precision('high')
    cli_main()