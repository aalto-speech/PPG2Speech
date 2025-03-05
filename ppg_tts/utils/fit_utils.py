import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import os

def WarmupCosineAnnealing(step, warmup_steps, total_steps, min_lr_factor=0.1):
    """
    Returns a multiplicative factor for the initial learning rate.
    
    - For step < warmup_steps: linear warmup from 0 to 1.
    - For warmup_steps <= step: cosine decay from 1 to min_lr_factor.
    """
    if step < warmup_steps:
        # Linear warmup: factor goes from 0 to 1.
        return float(step) / float(warmup_steps)
    else:
        # Cosine annealing: compute progress from 0 to 1.
        progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
        # Cosine annealing: at progress=0, factor=1; at progress=1, factor=min_lr_factor.
        return min_lr_factor + 0.5 * (1.0 - min_lr_factor) * (1.0 + math.cos(math.pi * progress))

        
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
