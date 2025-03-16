import json
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
        '--edited_ppg',
        type=str,
    )

    parser.add_argument(
        '--synthesized_ppg',
        type=str
    )

    parser.add_argument(
        '--edit_json',
        type=str,
    )

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    edit_path = Path(args.edit_json).parent

    logger.add(f"{edit_path.as_posix()}/ppg_evaluate.log", rotation='200 MB')

    # Load data
    edited_ppg_dict = load_scp(args.edited_ppg)
    synthesized_ppg_dict = load_scp(args.synthesized_ppg)

    with open(args.edit_json, 'r') as reader:
        edit_details = json.load(reader)

    logger.info("Start Evaluating Jensen-Shannon Divergence and Wasserstein Distance.")

    num_utt = 0
    num_frames = 0
    total_jsd = 0
    total_wass = 0
    for key in synthesized_ppg_dict:
        source_edit_ppg = edited_ppg_dict[key]
        synthesize_ppg = synthesized_ppg_dict[key]

        edited_region = edit_details[key]["edit_region"]

        region_slice = slice(edited_region[0], edited_region[1], 1)

        # Compare ppgs
        jsd = jensenshannon(source_edit_ppg[region_slice], synthesize_ppg[region_slice], axis=-1)
        wasserstein = wasserstein_distance_nd(source_edit_ppg[region_slice], synthesize_ppg[region_slice])

        logger.info(f"{key}, frame-level jensen-shannon divergence: {jsd.mean()}, wasserstein distance: {wasserstein}")

        num_utt += 1
        num_frames += (edited_region[1] - edited_region[0])
        total_jsd += jsd.sum()
        total_wass += wasserstein

    logger.info(
        f"Inference done. Average frame-level jensen-shannon divergence: {total_jsd / num_frames}"
    )

    logger.info(
        f"Inference done. Average wasserstein distance: {total_wass / num_utt}"
    ) 
