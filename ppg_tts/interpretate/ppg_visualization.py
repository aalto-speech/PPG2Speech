import sys
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from kaldiio import load_scp

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

def visualize_multiple_ppg(ppg_list, titles=None, figsize=(20, 12), save_path='./figure.png', 
                           y_labels_map=None, highlight_ranges=None):
    """
    Visualize multiple Phonetic Posteriograms (PPGs) in one graph.

    Parameters:
    - ppg_list: A list of PPG tensors, each of shape (T, C).
    - titles: A list of titles for each subplot (default: None).
    - figsize: Size of the entire figure (default: (20, 12)).
    - save_path: Path to save the figure (default: './figure.png').
    - y_labels_map: Optional dictionary mapping integer indices to string labels for y-axis.
    - highlight_range: Optional tuple (start, end) to highlight a specific time range with a red box.
    """
    num_ppg = len(ppg_list)
    if titles is None:
        titles = [f"Setting {i+1}" for i in range(num_ppg)]
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, math.ceil(num_ppg / 2), figsize=figsize)  # 2 rows, dynamic columns
    axes = axes.ravel()  # Flatten the axes array for easy indexing
    
    # Plot each PPG in a subplot
    for i, (ppg, title) in enumerate(zip(ppg_list, titles)):
        ax = axes[i]
        im = ax.imshow(ppg.T, aspect='auto', cmap='viridis', origin='lower')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Time Steps", fontsize=12)
        
        if y_labels_map:
            ax.set_yticks(range(len(y_labels_map)))
            ax.set_yticklabels([y_labels_map.get(idx, str(idx)) for idx in range(len(y_labels_map))], fontsize=10)
            ax.yaxis.set_tick_params(labelsize=10)
        else:
            ax.set_ylabel("Phoneme Classes", fontsize=12)
        
        fig.colorbar(im, ax=ax, label='Intensity', fraction=0.046, pad=0.04)
        
        # Highlight the specified range with a red rectangle
        if highlight_ranges:
            start, end = highlight_ranges[i]
            width = end - start
            height = ppg.shape[1]
            rect = patches.Rectangle((start, 0), width, height, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
    
    # Hide unused subplots
    for i in range(num_ppg, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1, wspace=0.3, hspace=0.4)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()



if __name__ == '__main__':
    key = sys.argv[1]

    label_map = {
        "<eps>": 0,
        "SIL": 1,
        "SPN": 2,
        "a": 3,
        "b": 4,
        "c": 5,
        "d": 6,
        "e": 7,
        "f": 8,
        "g": 9,
        "h": 10,
        "i": 11,
        "j": 12,
        "k": 13,
        "l": 14,
        "m": 15,
        "n": 16,
        "o": 17,
        "p": 18,
        "q": 19,
        "r": 20,
        "s": 21,
        "t": 22,
        "u": 23,
        "v": 24,
        "w": 25,
        "x": 26,
        "y": 27,
        "z": 28,
        "ä": 29,
        "å": 30,
        "ö": 31,
    }

    kaldi_ppgs = load_scp('data/spk_sanity/ppg.scp')
    edit_ppgs = load_scp('exp6_kaldi-ppgV2_conformer_transformer_4_mid2/editing_spk_sanity_rule_based/ppg.scp')
    synthesized_ppgs = load_scp('exp6_kaldi-ppgV2_conformer_transformer_4_mid2/editing_spk_sanity_rule_based/wav_hifigan/kaldi_dataset/ppg.scp')
    baseline_ppgs = load_scp('exp6_kaldi-ppgV2_conformer_transformer_4_mid2/editing_spk_sanity_rule_based/wav_baseline_hifigan/kaldi_dataset/ppg.scp')

    baseline_gt_ppgs = load_scp('eval_baseline/kaldi_dataset/ppg.scp')

    with open("exp6_kaldi-ppgV2_conformer_transformer_4_mid2/editing_spk_sanity_rule_based/edits.json", 'r') as reader:
        edits_json = json.load(reader)

    with open("exp6_kaldi-ppgV2_conformer_transformer_4_mid2/editing_spk_sanity_rule_based/matcha_edits.json", 'r') as reader:
        matcha_json = json.load(reader)

    kaldi_example = kaldi_ppgs[key]

    edit = edit_ppgs[key]

    synthe = synthesized_ppgs[key]

    baseline_gt = baseline_gt_ppgs[key]

    ppgs = [kaldi_example[:, :32], edit[:, :32], synthe[:, :32], baseline_ppgs[key][:, :32], baseline_gt[:, :32]]
    titles = ['original ppg', 'ppg after editing', 'synthesized ppg', 'ppg from tts baseline', 'ppg from tts baseline without editing']

    highlights = [edits_json[key]["edit_region"], edits_json[key]["edit_region"], 
                  edits_json[key]["edit_region"], matcha_json[key]["edit_region"],
                  [matcha_json[key]["edit_region"][0], matcha_json[key]["edit_region"][1]]]

    visualize_multiple_ppg(
        ppgs, titles, 
        save_path=f"exp6_kaldi-ppgV2_conformer_transformer_4_mid2/editing_spk_sanity_rule_based/{key}.png",
        y_labels_map={v: k for k, v in label_map.items()},
        highlight_ranges=highlights,
    )
