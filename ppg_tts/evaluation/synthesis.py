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

    parser.add_argument(
        '--rule_based_edit',
        action='store_true',
        default=False,
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
        flag = 'rule_based' if args.rule_based_edit else ''
        mel_save_dir = exp_dir / f"editing_{dirname}_{flag}/mel"
    else:
        mel_save_dir = exp_dir / f"mel_{dirname}"

    logger.add(f"{mel_save_dir}/synthesis.log", rotation='200 MB')

    device = args.device if torch.cuda.is_available() else 'cpu'

    logger.info(f"Load {args.model_class} checkpoint from {args.ckpt}, device is {device}")
    model_cls = import_obj_from_string(args.model_class)
    model, diff_steps, temperature = load_model(model_cls, args.ckpt, device)
    model = model.to(device)
    exp_dir = Path(args.ckpt).parent.parent

    logger.info(f"Load testset from {args.data_dir}")

    testset = ExtendDataset(
        data_dir=args.data_dir,
        no_ctc=False,
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
        edit_path = Path(mel_save_dir).parent
        editor = PPGEditor(args.phonemes)
        ppg_writer = WriteHelper(f'ark,scp:{edit_path.as_posix()}/ppg.ark,{edit_path.as_posix()}/ppg.scp')
        text_writer = open(f"{edit_path.as_posix()}/text", 'w', encoding='utf-8')
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
                new_ppg, (new_text, pos_in_str), region = editor.edit_ppg(
                    testdata['ppg'].squeeze(0).numpy(), text=text,
                    is_rule_based=args.rule_based_edit,
                )

                ppg_writer(source_key, new_ppg)
                text_writer.write(f"{source_key} {new_text}\n")

                logger.info(f"Editing {source_key}: [{text}] -> [{new_text}], frame range {region}")

                edits_json[source_key] = {
                    'origin_text': text,
                    'new_text': new_text,
                    'edit_region': region,
                    'pos_in_str': pos_in_str,
                }

                ppg = torch.from_numpy(new_ppg).unsqueeze(0)
            else:
                ppg = testdata['ppg'].to(device)

            pred_mel = model.synthesis(
                x=ppg.to(device),
                x_mask=testdata['ppg_mask'].to(device),
                spk_emb=target_spk_emb.unsqueeze(0).to(device),
                pitch_target=target_pitch.to(device),
                v_flag=testdata['v_flag'].to(device),
                mel_mask=testdata['mel_mask'].to(device),
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

        with open(f"{edit_path.as_posix()}/edits.json", 'w', encoding='utf-8') as writer:
            json.dump(
                edits_json,
                writer,
                indent=4,
                ensure_ascii=False,
            )
