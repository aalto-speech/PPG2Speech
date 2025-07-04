import os
import torch
import torchaudio
import utmosv2
from torchmetrics.functional.audio.dnsmos \
    import deep_noise_suppression_mean_opinion_score
from loguru import logger
from ..utils import build_parser

def read_wav_scp(scp: str):
    with open(scp, "r") as reader:
        lines = reader.readlines()

    d = {line.strip(' \n').split()[0]: line.strip(' \n').split()[1] for line in lines}

    for k, v in d.items():
        x, sr = torchaudio.load(v)
        yield x, sr, k

def read_wav(wav_dir: str):
    wavs = os.listdir(wav_dir)

    wavs = [os.path.join(wav_dir, wav) for wav in wavs if "generated_e2e.wav" in wav]

    for wav in wavs:
        key = wav.split('/')[-1].replace('_generated_e2e.wav', '')
        x, sr = torchaudio.load(wav)

        yield x, sr, key

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    logger.add(
        f"{args.flip_wav_dir}/logs/mos.log",
        rotation='200 MB'
    )

    average_score = torch.zeros((4,), device='cpu')

    # for wav, sr, key in read_wav_scp('/scratch/work/liz32/ppg_tts/data/spk_sanity/wav.scp'):
    for i, (wav, sr, key) in enumerate(read_wav(args.flip_wav_dir)):
        score = deep_noise_suppression_mean_opinion_score(
            preds=wav.to('cpu'),
            fs=sr,
            personalized=False,
            num_threads=8,
            device='cpu',
        )

        logger.info(
            f'{key}: p808_mos {score[0][0]}, mos_sig {score[0][1]}, '
            f'mos_bak {score[0][2]}, mos_ovr {score[0][3]}')
        average_score += (score.squeeze() - average_score) / (i + 1)
    
    logger.info(f'The average scores: p808_mos {average_score[0]}, '
                f'mos_sig {average_score[1]}, mos_bak {average_score[2]}, '
                f'mos_ovr {average_score[3]}')
    

