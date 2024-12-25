import os
import torch
import torchaudio
from torchmetrics.functional.audio.dnsmos \
    import deep_noise_suppression_mean_opinion_score
from loguru import logger
from ..utils import build_parser


def read_wav(wav_dir: str):
    wavs = os.listdir(wav_dir)

    wavs = [os.path.join(wav_dir, wav) for wav in wavs if "generated_e2e" in wav]

    for wav in wavs:
        key = wav.split('/')[-1].replace('_generated_e2e.wav', '')
        x, sr = torchaudio.load(wav)

        yield x, sr, key

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    average_score = torch.zeros((4,))
    num_audio  = 0

    for wav, sr, key in read_wav(args.flip_wav_dir):
        score = deep_noise_suppression_mean_opinion_score(
            preds=wav,
            fs=sr,
            personalized=False,
            num_threads=8,
        )

        logger.info(
            f'{key}: p808_mos {score[0][0]}, mos_sig {score[0][1]}, '
            f'mos_bak {score[0][2]}, mos_ovr {score[0][3]}')
        average_score = average_score + score.squeeze()
        num_audio += 1
    
    average_score /= num_audio

    logger.info(f'The average scores: p808_mos {average_score[0]}, '
                f'mos_sig {average_score[1]}, mos_bak {average_score[2]}, '
                f'mos_ovr {average_score[3]}')
