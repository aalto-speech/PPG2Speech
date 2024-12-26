import os
import torch
import random
import json
import numpy as np
from loguru import logger
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Tuple
from ..dataset import ExtendDataset, PersoCollateFn
from ..models import VQVAEMatcha
from ..utils import build_parser

def replace_spk_emb(testset: ExtendDataset, curr_idx: int) -> Tuple[str, torch.Tensor]:
    random_idx = random.randint(0, len(testset) - 1)

    while random_idx == curr_idx:
        random_idx = random.randint(0, len(testset) - 1)

    return testset[random_idx]['key'], testset[random_idx]['spk_emb']

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    ckpt = torch.load(args.ckpt, map_location=args.device)

    logger.info(f"Load checkpoint from {args.ckpt}, device is {args.device}")

    with open(ckpt['hyper_parameters']['pitch_stats'], "r") as reader:
            pitch_stats = json.load(reader)

    model = VQVAEMatcha(
        **ckpt['hyper_parameters'],
        **pitch_stats,
    )

    weights = {k.replace('model.', ''): v for k, v in ckpt['state_dict'].items()}

    model.load_state_dict(weights)

    logger.info(f"Load testset from {args.data_dir}")

    testset = ExtendDataset(
        data_dir=args.data_dir,
    )

    with open(f"{args.data_dir}/picth_median_per_speaker.json", "r") as median_reader:
        speaker_median = json.load(median_reader)

    testloader = DataLoader(
        dataset=testset,
        batch_size=1,
        num_workers=4,
        collate_fn=PersoCollateFn,
        shuffle=False,
    )

    exp_dir = Path(args.ckpt).parent.parent
    if args.switch_speaker:
        mel_save_dir = exp_dir / "flip_generate_mel"
    else:
        mel_save_dir = exp_dir / "mel"

    os.makedirs(mel_save_dir, exist_ok=True)

    speaker_mapping = open(mel_save_dir / "speaker_mapping", "w")

    model.eval()

    for i, testdata in enumerate(testloader):
        # Inference mel spectrogram
        source_key = testdata['keys'][0]
        if args.switch_speaker:
            target_key, target_spk_emb = replace_spk_emb(testset=testset, curr_idx=i)
            
            # Pitch shift in inference
            curr_speaker = source_key.split('_')[0]
            target_speaker = target_key.split('_')[0]
            curr_median = speaker_median[curr_speaker]
            target_median = speaker_median[target_speaker]
            shifted_pitch = testdata['log_F0'] * target_median / curr_median
        else:
            target_key, target_spk_emb = source_key, testdata['spk_emb'].squeeze(0)
            shifted_pitch = testdata['log_F0']
        logger.info(f"generate {source_key} with speaker embedding from {target_key}")
        print(f"{source_key} {target_key}", file=speaker_mapping)

        pred_mel, _, _, _ = model.synthesis(
            x=testdata['ppg'],
            x_mask=testdata['ppg_mask'],
            spk_emb=target_spk_emb.unsqueeze(0),
            pitch_target=shifted_pitch,
            v_flag=testdata['v_flag'],
            mel_mask=testdata['mel_mask'],
            diff_steps=ckpt['hyper_parameters']['diff_steps'],
            temperature=ckpt['hyper_parameters']['temperature']
        )

        saved_mel = pred_mel.transpose(1,2).detach().cpu().numpy()

        # Save mel spectrogram
        np.save(f"{mel_save_dir}/{source_key}", saved_mel)

    speaker_mapping.close()
