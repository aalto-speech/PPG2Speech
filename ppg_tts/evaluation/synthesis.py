import argparse
import json
import os
import torch
import random
import numpy as np
from kaldiio import WriteHelper
from loguru import logger
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Tuple
from ..dataset import ExtendDataset, PersoCollateFn
from .evaluate_ppg.ppg_edit import PPGEditor
from ..utils import load_model, import_obj_from_string

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Argument parser for synthesis.")
    parser.add_argument("--data_dir", 
                        help="The data directory for Dataset. Should contains wav.scp and text for Perso. Download location for VCTK.",
                        default="./data")
    parser.add_argument("--device",
                        help="CPU/CUDA device to run",
                        default="cpu")
    parser.add_argument("--ckpt",
                        help="ckpt path for the TTS/VC model.")
    parser.add_argument('--switch_speaker',
                        help='whether to switch the speaker embedding during inference',
                        action='store_true')
    parser.add_argument(
        '--model_class',
        help='The path to the model class',
        type=str,
    )
    parser.add_argument(
        '--edit_ppg',
        action='store_true',
        default=False,
    )

    parser.add_argument(
        '--phonemes',
        type=str,
        default='data/spk_sanity/phones.txt'
    )

    return parser

def replace_spk_emb(testset: ExtendDataset, curr_idx: int) -> Tuple[str, torch.Tensor]:
    random_idx = random.randint(0, len(testset) - 1)

    while random_idx == curr_idx:
        random_idx = random.randint(0, len(testset) - 1)

    return testset[random_idx]['key'], testset[random_idx]['spk_emb']

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.edit_ppg and args.switch_speaker:
        raise ValueError("Can't synthesize with edited ppg while switch speaker identity")
    
    exp_dir = Path(args.ckpt).parent.parent
    dirname = os.path.basename(args.data_dir)
    
    if args.switch_speaker:
        mel_save_dir = exp_dir / f"flip_generate_mel_{dirname}"
    elif args.edit_ppg:
        mel_save_dir = exp_dir / f"edit_mel_{dirname}"
    else:
        mel_save_dir = exp_dir / f"mel_{dirname}"

    logger.add(f"{mel_save_dir}/synthesis.log", rotation='200 MB')

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

    os.makedirs(mel_save_dir, exist_ok=True)

    speaker_mapping = open(mel_save_dir / "speaker_mapping", "w")

    if args.edit_ppg:
        os.makedirs(f"{mel_save_dir.as_posix()}/editing", exist_ok=True)
        editor = PPGEditor(args.phonemes)
        ppg_writer = WriteHelper(f'ark,scp:{mel_save_dir.as_posix()}/editing/ppg.ark,{mel_save_dir.as_posix()}/editing/ppg.scp')
        text_writer = open(f"{mel_save_dir.as_posix()}/editing/text", 'w', encoding='utf-8')
        edits_json = {}

    with torch.inference_mode():
        for i, testdata in enumerate(testloader):
            # Inference mel spectrogram
            source_key = testdata['keys'][0]
            if args.switch_speaker:
                target_key, target_spk_emb = replace_spk_emb(testset=testset, curr_idx=i)
            else:
                target_key, target_spk_emb = source_key, testdata['spk_emb'].squeeze(0)
            target_pitch = testdata['log_F0']
            logger.info(f"Synthesize {source_key} with speaker embedding from {target_key}")
            print(f"{source_key} {target_key}", file=speaker_mapping)

            if args.edit_ppg:
                text = testset[i]['text']
                new_ppg, new_text, region = editor.edit_ppg(
                    testdata['ppg'].squeeze(0).numpy(), text=text
                )

                ppg_writer(source_key, new_ppg)
                text_writer.write(f"{source_key} {new_text}\n")

                logger.info(f"Editing {source_key}: [{text}] -> [{new_text}], frame range {region}")

                edits_json[source_key] = {
                    'origin_text': text,
                    'new_text': new_text,
                    'edit_region': region,
                }

                ppg = torch.from_numpy(new_ppg).unsqueeze(0)
            else:
                ppg = testdata['ppg'].to(args.device)

            pred_mel = model.synthesis(
                x=ppg.to(args.device),
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

    if args.edit_ppg:
        ppg_writer.close()
        text_writer.close()

        with open(f"{mel_save_dir.as_posix()}/editing/edits.json", 'w') as writer:
            json.dump(
                edits_json,
                writer,
                indent=4,
            )
