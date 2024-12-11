import torch
import numpy as np
import json
from collections import defaultdict
from loguru import logger
from ..utils import build_parser
from ..dataset import ExtendDataset

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    dataset = ExtendDataset(args.data_dir)

    logger.info("Get pitch median for each speaker")

    all_pitch_per_speaker = defaultdict(list)

    for data in dataset:
        key = data["key"]

        speaker = key.split('_')[0]

        pitch = data['log_F0']
        uv = data['v_flag']

        orig_pitch = pitch * uv

        orig_pitch = orig_pitch[orig_pitch > 0].tolist()

        all_pitch_per_speaker[speaker].extend(orig_pitch)

    median = {spk: np.median(pitches) for spk, pitches in all_pitch_per_speaker.items()}

    with open(f"{args.data_dir}/picth_median_per_speaker.json", "w") as f:
        json.dump(median, f, indent=4)

    logger.info(f"Extracting stats on log F0 and Energy to {args.data_dir}, in total {len(dataset)} utterances.")

    log_F0_min, log_F0_max, energy_min, energy_max = np.inf, -np.inf, np.inf, -np.inf

    for data in dataset:
        curr_log_F0_min = data["log_F0"].min().item()
        curr_log_F0_max = data["log_F0"].max().item()

        curr_energy_min = data["energy"].min().item()
        curr_energy_max = data["energy"].max().item()

        logger.info(f"At utterance {data['key']}, pitch min {curr_log_F0_min:.3f}, \
                    pitch max: {curr_log_F0_max:.3f}, \
                    energy min: {curr_energy_min:.3f}, \
                    energy max: {curr_energy_max:.3f}")

        if curr_log_F0_min < log_F0_min:
            log_F0_min = curr_log_F0_min

        if curr_log_F0_max > log_F0_max:
            log_F0_max = curr_log_F0_max

        if curr_energy_min < energy_min:
            energy_min = curr_energy_min

        if curr_energy_max > energy_max:
            energy_max = curr_energy_max

    with open(f"{args.data_dir}/stats.json", "w") as f:
        json.dump({
            "pitch_max": log_F0_max,
            "pitch_min": log_F0_min,
            "energy_max": energy_max,
            "energy_min": energy_min
        }, f, indent=4)
