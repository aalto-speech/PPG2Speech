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
from ..utils import build_parser, load_model, import_obj_from_string
from ..models import VQVAEMatcha, PPGMatcha

def replace_spk_emb(testset: ExtendDataset, curr_idx: int) -> Tuple[str, torch.Tensor]:
    random_idx = random.randint(0, len(testset) - 1)

    while random_idx == curr_idx:
        random_idx = random.randint(0, len(testset) - 1)

    return testset[random_idx]['key'], testset[random_idx]['spk_emb']

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    logger.info(f"Load {args.model_class} checkpoint from {args.ckpt}, device is {args.device}")
    model_cls = import_obj_from_string(args.model_class)
    model, diff_steps, temperature = load_model(model_cls, args.ckpt, args.device)
    model = model.to(args.device)
    exp_dir = Path(args.ckpt).parent.parent

    # sparse_method, sparse_coeff = exp_dir.as_posix().split('_')[-2:]

    # match sparse_method:
    #     case 'topk': sparse_coeff = int(sparse_coeff)
    #     case 'percentage': sparse_coeff = float(sparse_coeff)
    #     case _: raise ValueError(f"Failed to parse sparse method from {exp_dir}")

    logger.info(f"Load testset from {args.data_dir}")

    testset = ExtendDataset(
        data_dir=args.data_dir,
        no_ctc=False,
        # ppg_sparse=sparse_method,
        # sparse_coeff=sparse_coeff,
    )

    testloader = DataLoader(
        dataset=testset,
        batch_size=1,
        num_workers=4,
        collate_fn=PersoCollateFn,
        shuffle=False,
    )

    dirname = os.path.dirname(args.data_dir)

    if args.switch_speaker:
        mel_save_dir = exp_dir / f"flip_generate_mel_{dirname}"
    else:
        mel_save_dir = exp_dir / f"mel_{dirname}"

    os.makedirs(mel_save_dir, exist_ok=True)

    speaker_mapping = open(mel_save_dir / "speaker_mapping", "w")

    with torch.inference_mode():
        for i, testdata in enumerate(testloader):
            # Inference mel spectrogram
            source_key = testdata['keys'][0]
            if args.switch_speaker:
                target_key, target_spk_emb = replace_spk_emb(testset=testset, curr_idx=i)
            else:
                target_key, target_spk_emb = source_key, testdata['spk_emb'].squeeze(0)
            target_pitch = testdata['log_F0']
            logger.info(f"generate {source_key} with speaker embedding from {target_key}")
            print(f"{source_key} {target_key}", file=speaker_mapping)

            pred_mel = model.synthesis(
                x=testdata['ppg'].to(args.device),
                x_mask=testdata['ppg_mask'].to(args.device),
                spk_emb=target_spk_emb.unsqueeze(0).to(args.device),
                pitch_target=target_pitch.to(args.device),
                v_flag=testdata['v_flag'].to(args.device),
                mel_mask=testdata['mel_mask'].to(args.device),
                diff_steps=diff_steps,
                temperature=temperature,
            )

            saved_mel = pred_mel.transpose(1,2).cpu().numpy()

            # Save mel spectrogram
            np.save(f"{mel_save_dir}/{source_key}", saved_mel)

    speaker_mapping.close()
