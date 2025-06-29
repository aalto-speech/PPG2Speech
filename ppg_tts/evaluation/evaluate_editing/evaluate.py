import os
import json
from argparse import ArgumentParser
from fastdtw import fastdtw
from kaldiio import load_scp
from pathlib import Path
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

    parser.add_argument(
        '--matcha_mfa_align',
        type=str,
        default=None,
    )

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    if args.matcha_aligned_edits and args.matcha_mfa_align:
        raise ValueError("Please provide either matcha_aligned_edits or matcha_mfa_align, not both.")
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

    logger.info(f"Start Evaluating Jensen-Shannon Divergence"
                f"for {Path(args.synthesized_ppg).parent.parent.as_posix()}.")

    num_frames = 0
    average_jsd = 0
    for i, key in enumerate(synthesized_ppg_dict):
        source_edit_ppg = edited_ppg_dict[key]
        synthesize_ppg = synthesized_ppg_dict[key]

        edited_region = edit_details[key]["edit_region"]

        region_slice = slice(edited_region[0], edited_region[1], 1)

        source_region = source_edit_ppg[region_slice]

        num_invalid = 0

        if args.matcha_aligned_edits is not None:
            matcha_region = slice(
                matcha_aligned_edits[key]["edit_region"][0],
                matcha_aligned_edits[key]["edit_region"][1],
                1
            )
            synthesized_editing = synthesize_ppg[matcha_region]
            matcha_frames = (matcha_aligned_edits[key]["edit_region"][1] - matcha_aligned_edits[key]["edit_region"][0])
            num_frames = (edited_region[1] - edited_region[0])
        elif args.matcha_mfa_align is not None:
            spk = key.split("_")[0]
            alignment_json_path = os.path.join(
                args.matcha_mfa_align,
                spk,
                f"{key}.json"
            )

            with open(alignment_json_path, 'r') as reader:
                alignment_data = json.load(reader)['tiers']

            if isinstance(edit_details[key]['pos_in_str'], list):
                prefix_until = edit_details[key]['pos_in_str'][0]
            else:
                prefix_until = edit_details[key]['pos_in_str']

            prefix = edit_details[key]['new_text'][:prefix_until]
            entry_idx = prefix_until - prefix.count(" ")

            if isinstance(edit_details[key]['pos_in_str'], list):
                entry = alignment_data['phones']['entries'][entry_idx]
                second_entry = alignment_data['phones']['entries'][entry_idx + 1]
                start_frame, end_frame = int(entry[0] * 100), int(second_entry[1] * 100)
            else:
                entry = alignment_data['phones']['entries'][entry_idx]
                start_frame, end_frame = int(entry[0] * 100), int(entry[1] * 100)
            synthesized_editing = synthesize_ppg[start_frame:end_frame]
            num_frames = (edited_region[1] - edited_region[0])
        else:
            synthesized_editing = synthesize_ppg[region_slice]
            num_frames = (edited_region[1] - edited_region[0])

        jsd, _ = fastdtw(source_region, synthesized_editing[:, :32], 10, jensenshannon)

        #! Filter Possibly invalid results
        if jsd == 0.0:
            logger.warning(f"{key}: invalid cost computation")
            num_invalid += 1
            continue
        else:
            logger.info(f"{key}, frame-level jensen-shannon divergence: {jsd / num_frames}")
        average_jsd += ((jsd / num_frames) - average_jsd) / (i - num_invalid + 1)

    logger.info(
        f"Inference done. Average frame-level jensen-shannon divergence: {average_jsd}"
    )
