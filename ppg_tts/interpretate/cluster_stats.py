import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
from matplotlib.cm import get_cmap

def make_bar_plot(data: defaultdict, fig_name: str):
    categories = list(data.keys())
    sub_keys = sorted(set(k for d in data.values() for k in d.keys()))

    # Creating a 2D array where each row represents a category and each column a sub-key
    values = np.zeros((len(categories), len(sub_keys)))

    num_colors = len(sub_keys)
    colormap = get_cmap("tab20") if num_colors <= 20 else get_cmap("gist_rainbow")
    colors = [colormap(i / num_colors) for i in range(num_colors)]

    for i, category in enumerate(categories):
        for j, sub_key in enumerate(sub_keys):
            values[i, j] = data[category].get(sub_key, 0)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(categories))  # One bar per category
    bottom = np.zeros(len(categories))  # Initialize bottom to 0 for stacking

    # Stacked bars
    for j, sub_key in enumerate(sub_keys):
        ax.bar(
            x,
            values[:, j],
            bottom=bottom,
            color=colors[j],
            label=f"{sub_key}"
        )
        bottom += values[:, j]  # Update bottom for stacking

    # Adding labels and legend
    ax.set_xlabel("Categories")
    ax.set_ylabel("Values")
    ax.set_title("Stacked Bar Chart of Sub Keys by Category")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45)
    ax.legend(title="Sub-keys", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    plt.savefig(f'ppg_tts/interpretate/{fig_name}', dpi=300)

if __name__ == '__main__':
    token2cluster = 'ppg_tts/interpretate/cluster.json'

    with open(token2cluster, 'r') as reader:
        d = json.load(reader)

    cluster2token = defaultdict(Counter)

    for token in d:
        for cluster in d[token]:
            cluster2token[cluster][token] = d[token][cluster]

    make_bar_plot(d, 'token2cluster')

    make_bar_plot(cluster2token, 'cluster2token')