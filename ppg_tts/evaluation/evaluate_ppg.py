import numpy as np
import torch
import yaml
from argparse import ArgumentParser
from collections import OrderedDict
from kaldiio import load_scp
from pathlib import Path
from scipy.stats import wasserstein_distance_nd
from scipy.spatial.distance import jensenshannon
from torch.nn.functional import softmax
from transformers import (
    Wav2Vec2Config,
    Wav2Vec2ForAudioFrameClassification,
)
from loguru import logger

from ..dataset.generalDataset import sparse_topK, sparse_topK_percent
from .evaluate_wer import read_wav_text

def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        '--wav_dir',
        '-w',
        type=str,
    )

    parser.add_argument(
        '--data_dir',
        '-d',
        type=str
    )

    parser.add_argument(
        '--ppg_ckpt',
        '-c',
        type=str
    )

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    # Load data
    logger.info(f'Using wav dir {args.wav_dir}, data dir {args.data_dir}')
    text = Path(args.data_dir) / 'text'
    ppg_scp = Path(args.data_dir) / 'ppg_nn_lsm0.2.scp'

    ppg_dict = load_scp(ppg_scp.as_posix())

    # Determine devices
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    logger.info (f"Inference using device {device}")

    # Setup sparse function
    exp_conf_path = Path(args.wav_dir).parent / 'config.yaml'
    
    with open(exp_conf_path, 'r') as reader:
        exp_conf = yaml.safe_load(reader)

    match exp_conf['data']['init_args']['ppg_sparse']:
        case 'topk': sparse_func = sparse_topK
        case 'percentage': sparse_func = sparse_topK_percent
        case _: raise ValueError(
            f"{exp_conf['data']['init_args']['ppg_sparse']} is not a suppprt ppg sparse method"
        )

    sparse_coeff = exp_conf['data']['init_args']['sparse_coeff']
    logger.info(f"Use {sparse_func.__name__} function, sparse coeff {sparse_coeff}")

    # Load model weights
    model_conf_path = Path(args.ppg_ckpt).parent / "model_conf.json"
    model_conf = Wav2Vec2Config.from_json_file(model_conf_path.as_posix())

    model = Wav2Vec2ForAudioFrameClassification(model_conf)

    state_dict = torch.load(args.ppg_ckpt, map_location=device)['model'] \
        if 'final' not in args.ppg_ckpt else \
        torch.load(args.ppg_ckpt, map_location=device)
    
    if 'final' in args.ppg_ckpt or 'ema' in args.ppg_ckpt:
        new_state_dict = OrderedDict()
        for k in state_dict:
            new_state_dict[k.replace('module.', '')] = state_dict[k]
        state_dict = new_state_dict
        state_dict.pop('n_averaged')

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    logger.warning(f"Loading model: Missing keys {missing}, Unexpected keys {unexpected}")

    logger.info("Start Evaluation")
    num_utt = 0
    num_frames = 0
    total_jsd = 0
    total_wass = 0
    with torch.inference_mode():
        for utterance in read_wav_text(args.wav_dir, text.as_posix()):
            wav = utterance[0].to(device)
            key = utterance[2]
            logits = model.forward(wav).logits

            target_logits = ppg_dict[key]

            #! Do Some Evaluation here
            source_prob = softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            target_prob = softmax(target_logits, dim=-1).squeeze(0).cpu().numpy()

            jsd = jensenshannon(source_prob, target_prob, axis=-1)
            wasserstein = wasserstein_distance_nd(source_prob, target_prob)

            logger.info(f"{key}, frame-level jensen-shannon divergence: {jsd.mean()}, wasserstein distance: {wasserstein}")

            num_utt += 1
            num_frames += source_prob.shape[0]
            total_jsd += jsd.sum()
            total_wass += wasserstein

        logger.info(
            f"Inference done. Average frame-level jensen-shannon divergence: {total_jsd / num_frames}"
        )

        logger.info(
            f"Inference done. Average wasserstein distance: {total_wass / num_utt}"
        )
