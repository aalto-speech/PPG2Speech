import sys
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import softmax
from kaldiio import load_scp
from ..dataset.generalDataset import sparse_topK, sparse_topK_percent

def visualize_multiple_ppg(ppg_list, titles=None, figsize=(20, 12), save_path='./figure.png', y_labels_map=None):
    """
    Visualize multiple Phonetic Posteriograms (PPGs) in one graph.

    Parameters:
    - ppg_list: A list of PPG tensors, each of shape (T, C).
    - titles: A list of titles for each subplot (default: None).
    - figsize: Size of the entire figure (default: (20, 12)).
    - save_path: Path to save the figure (default: './figure.png').
    - y_labels_map: Optional dictionary mapping integer indices to string labels for y-axis.
    """
    num_ppg = len(ppg_list)
    if titles is None:
        titles = [f"Setting {i+1}" for i in range(num_ppg)]
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, math.ceil(len(ppg_list) / 2), figsize=figsize)  # 2 rows, dynamic columns
    axes = axes.ravel()  # Flatten the axes array for easy indexing
    
    # Plot each PPG in a subplot
    for i, (ppg, title) in enumerate(zip(ppg_list, titles)):
        ax = axes[i]
        im = ax.imshow(ppg.T, aspect='auto', cmap='viridis', origin='lower')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Time Steps", fontsize=12)
        
        if y_labels_map:
            ax.set_yticks(range(32))
            ax.set_yticklabels([y_labels_map.get(idx, str(idx)) for idx in range(32)], fontsize=10)
            ax.yaxis.set_tick_params(labelsize=10)
        else:
            ax.set_ylabel("Phoneme Classes", fontsize=12)
        
        fig.colorbar(im, ax=ax, label='Intensity', fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for i in range(num_ppg, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1, wspace=0.3, hspace=0.4)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == '__main__':
    key = sys.argv[1]

    kaldi_ppgs = load_scp('data/spk_sanity/ppg.scp')
    nn_ppgs = load_scp('/scratch/elec/t412-speechsynth/DATA/fin-mix/test/ppg_nn_lsm0.2.scp')

    kaldi_example = kaldi_ppgs[key]

    nn_example = torch.from_numpy(nn_ppgs[key].copy())

    original_nn = softmax(nn_example, dim=-1)

    ppgs = [kaldi_example]
    titles = ['kaldi ppg']

    ppgs.append(original_nn.numpy())
    titles.append('original nn ppg')

    ppgs.append(sparse_topK(nn_example.unsqueeze(0), 2).squeeze(0).numpy())
    titles.append('nn ppg with top2 phonemes')

    ppgs.append(sparse_topK(nn_example.unsqueeze(0), 3).squeeze(0).numpy())
    titles.append('nn ppg with top3 phonemes')

    ppgs.append(sparse_topK(nn_example.unsqueeze(0), 4).squeeze(0).numpy())
    titles.append('nn ppg with top4 phonemes')

    ppgs.append(sparse_topK_percent(nn_example.unsqueeze(0), 0.85).squeeze(0).numpy())
    titles.append('nn ppg with top 85% prob phonemes')

    ppgs.append(sparse_topK_percent(nn_example.unsqueeze(0), 0.9).squeeze(0).numpy())
    titles.append('nn ppg with top 90% prob phonemes')

    ppgs.append(sparse_topK_percent(nn_example.unsqueeze(0), 0.95).squeeze(0).numpy())
    titles.append('nn ppg with top 95% prob phonemes')

    visualize_multiple_ppg(ppgs, titles, save_path=f"ppg_tts/interpretate/{key}_lsm0.2.png")
