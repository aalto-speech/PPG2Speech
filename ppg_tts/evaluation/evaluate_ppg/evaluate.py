import json
import numpy as np
from argparse import ArgumentParser
from fastdtw import fastdtw
from kaldiio import load_scp
from pathlib import Path
from scipy.stats import wasserstein_distance
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

    parser.add_argument(
        '--matcha_aligned_edits',
        type=str,
        default=None,
    )

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    log_path = Path(args.synthesized_ppg).parent / 'log'

    logger.add(f"{log_path.as_posix()}/ppg_evaluate.log", rotation='200 MB')

    # Load data
    edited_ppg_dict = load_scp(args.edited_ppg)
    synthesized_ppg_dict = load_scp(args.synthesized_ppg)

    with open(args.edit_json, 'r') as reader:
        edit_details = json.load(reader)

    if args.matcha_aligned_edits is not None:
        with open(args.matcha_aligned_edits, 'r') as reader:
            matcha_aligned_edits = json.load(reader)

    logger.info(f"Start Evaluating Jensen-Shannon Divergence and Wasserstein Distance"
                f"for {Path(args.synthesized_ppg).parent.parent.as_posix()}.")

    num_frames = 0
    average_jsd = 0
    average_wass = 0
    for i, key in enumerate(synthesized_ppg_dict):
        source_edit_ppg = edited_ppg_dict[key]
        synthesize_ppg = synthesized_ppg_dict[key]

        edited_region = edit_details[key]["edit_region"]

        region_slice = slice(edited_region[0], edited_region[1], 1)

        source_region = source_edit_ppg[region_slice]

        if args.matcha_aligned_edits is not None:
            matcha_region = slice(
                matcha_aligned_edits[key]["edit_region"][0],
                matcha_aligned_edits[key]["edit_region"][1],
                1
            )
            synthesized_editing = synthesize_ppg[matcha_region]
            matcha_frames = (matcha_aligned_edits[key]["edit_region"][1] - matcha_aligned_edits[key]["edit_region"][0])
            source_frames = (edited_region[1] - edited_region[0])
            num_frames = max(matcha_frames, source_frames)
        else:
            synthesized_editing = synthesize_ppg[region_slice]
            num_frames = (edited_region[1] - edited_region[0])

        jsd, _ = fastdtw(source_region, synthesized_editing, 10, jensenshannon)

        wasserstein, _ = fastdtw(source_region, synthesized_editing, 10, wasserstein_distance)

        logger.info(f"{key}, frame-level jensen-shannon divergence: {jsd}, wasserstein distance: {wasserstein}")

        average_jsd += ((jsd / num_frames) - average_jsd) / (i + 1)
        average_wass += ((wasserstein / num_frames) - average_jsd) / (i + 1)

    logger.info(
        f"Inference done. Average frame-level jensen-shannon divergence: {average_jsd / num_frames}"
    )

    logger.info(
        f"Inference done. Average wasserstein distance: {average_wass / num_frames}"
    ) 
