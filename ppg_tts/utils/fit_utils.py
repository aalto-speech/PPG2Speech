import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl

class RunTestOnFitEndCallback(pl.Callback):
    def on_fit_end(self, trainer, pl_module):
        trainer.test(ckpt_path='best',
                     datamodule=trainer.datamodule)
        
def save_figure_to_numpy(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

        
def plot_tensor_wandb(t: torch.Tensor):
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(t, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data

def plot_mel(mel: np.ndarray, path: str, key: str):
    if not os.path.exists(path):
        os.makedirs(path)

    np.save(f"{path}/{key}", mel)

    plt.figure(figsize=(10, 4))
    plt.imshow(mel[0], aspect="auto", origin="lower", cmap="magma",
               extent=[0, mel.shape[0], 0, mel.shape[1]])
    plt.title(f"Predicted Mel Spectrogram for {key}")
    plt.xlabel("Time")
    plt.ylabel("Mel Frequency")

    plt.savefig(f"{path}/{key}.png", format="png", dpi=300, bbox_inches="tight")
    plt.close() 
