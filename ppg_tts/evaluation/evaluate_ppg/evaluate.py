import numpy as np
from argparse import ArgumentParser
from kaldiio import load_scp
from pathlib import Path
from scipy.stats import wasserstein_distance_nd
from scipy.spatial.distance import jensenshannon
from loguru import logger

def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        '--tts_baseline_ppg',
        type=str,
    )

    parser.add_argument(
        '--ppg',
        type=str
    )

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    edit_path = Path(args.ppg).parent.parent

    logger.add(f"{edit_path.as_posix()}/ppg_evaluate.log", rotation='200 MB')

    # Load data
    tts_base_dict = load_scp(args.tts_baseline_ppg)
    ppgs_dict = load_scp(args.ppg)

    logger.info("Start Evaluating Jensen-Shannon Divergence and Wasserstein Distance.")

    num_utt = 0
    num_frames = 0
    total_jsd = 0
    total_wass = 0
    for key, ppg in ppgs_dict:
        tts_base_ppg = tts_base_dict[key]

        # Compare ppgs
        jsd = jensenshannon(tts_base_ppg, ppg, axis=-1)
        wasserstein = wasserstein_distance_nd(tts_base_ppg, ppg)

        logger.info(f"{key}, frame-level jensen-shannon divergence: {jsd.mean()}, wasserstein distance: {wasserstein}")

        num_utt += 1
        num_frames += ppg.shape[0]
        total_jsd += jsd.sum()
        total_wass += wasserstein

    logger.info(
        f"Inference done. Average frame-level jensen-shannon divergence: {total_jsd / num_frames}"
    )

    logger.info(
        f"Inference done. Average wasserstein distance: {total_wass / num_utt}"
    )
    
