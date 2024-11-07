import os
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl

class RunTestOnFitEndCallback(pl.Callback):
    def on_fit_end(self, trainer, pl_module):
        trainer.test(ckpt_path='best',
                     datamodule=trainer.datamodule)

def plot_mel(mel: np.ndarray, path: str, key: str):
    if not os.path.exists(path):
        os.makedirs(path)

    np.save(f"{path}/{key}", mel)

    plt.figure(figsize=(10, 4))
    plt.imshow(mel.T, aspect="auto", origin="lower", cmap="magma")
    plt.title(f"Predicted Mel Spectrogram for {key}")
    plt.xlabel("Time")
    plt.ylabel("Mel Frequency")

    plt.savefig(f"{path}/{key}.png", format="png", dpi=300, bbox_inches="tight")
    plt.close() 
