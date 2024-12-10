import os
import torch
import random
import numpy as np
from loguru import logger
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Tuple
from ..dataset import ExtendDataset, PersoCollateFn
from ..models import ConformerMatchaTTS
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

    model = ConformerMatchaTTS(
        ppg_dim=ckpt['hyper_parameters']['ppg_dim'],
        encode_dim=ckpt['hyper_parameters']['encode_dim'],
        encode_heads=ckpt['hyper_parameters']['encode_heads'],
        encode_layers=ckpt['hyper_parameters']['encode_layers'],
        encode_ffn_dim=ckpt['hyper_parameters']['encode_ffn_dim'],
        encode_kernel_size=ckpt['hyper_parameters']['encode_kernel_size'],
        spk_emb_size=ckpt['hyper_parameters']['spk_emb_size'],
        decoder_num_block=ckpt['hyper_parameters']['decoder_num_block'],
        decoder_num_mid_block=ckpt['hyper_parameters']['decoder_num_mid_block'],
        dropout=ckpt['hyper_parameters']['dropout'],
        target_dim=ckpt['hyper_parameters']['target_dim'],
        sigma_min=ckpt['hyper_parameters']['sigma_min'],
        transformer_type=ckpt['hyper_parameters']['transformer_type'],
        no_ctc=ckpt['hyper_parameters']['no_ctc'],
    )

    weights = {k.replace('model.', ''): v for k, v in ckpt['state_dict'].items()}

    model.load_state_dict(weights)

    logger.info(f"Load testset from {args.data_dir}")

    testset = ExtendDataset(
        data_dir=args.data_dir,
    )

    testloader = DataLoader(
        dataset=testset,
        batch_size=1,
        num_workers=4,
        collate_fn=PersoCollateFn,
        shuffle=False,
    )

    exp_dir = Path(args.ckpt).parent.parent
    mel_save_dir = exp_dir / "flip_generate_mel"

    os.makedirs(mel_save_dir, exist_ok=True)

    speaker_mapping = open(mel_save_dir / "speaker_mapping", "w")

    for i, testdata in enumerate(testloader):
        # Inference mel spectrogram
        target_key, target_spk_emb = replace_spk_emb(testset=testset, curr_idx=i)
        logger.info(f"generate {testdata['keys'][0]} with speaker embedding from {target_key}")
        print(f"{testdata['keys'][0]} {target_key}", file=speaker_mapping)
        pred_mel = model.synthesis(
            x=testdata['ppg'],
            spk_emb=target_spk_emb.unsqueeze(0),
            pitch_target=testdata['log_F0'],
            v_flag=testdata['v_flag'],
            energy_length=testdata['energy_len'],
            mel_mask=testdata['mel_mask'],
            diff_steps=ckpt['hyper_parameters']['diff_steps'],
            temperature=ckpt['hyper_parameters']['temperature']
        )

        saved_mel = pred_mel.transpose(1,2).detach().cpu().numpy()

        # Save mel spectrogram
        np.save(f"{mel_save_dir}/{testdata['keys'][0]}", saved_mel)

    speaker_mapping.close()
