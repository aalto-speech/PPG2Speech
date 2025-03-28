import torchaudio
import penn
import pymcd
from functools import partial
from loguru import logger
from torch.nn.functional import pad
from typing import Tuple
from ..dataset import ExtendDataset
from ..utils import build_parser

compute_pitch = partial(
    penn.from_audio,
    sample_rate=22050,
    hopsize=(256 / 22050),
    fmin=40,
    fmax=700,
    center='zero',
    checkpoint=None,
    interp_unvoiced_at=0.2,
)

def compute_pitch_mae(audio1: str, audio2: str) -> Tuple[float, int]:
    x1, sr1 = torchaudio.load(audio1)
    x2, sr2 = torchaudio.load(audio2)

    x1 = torchaudio.functional.resample(
        waveform=x1,
        orig_freq=sr1,
        new_freq=22050,
    )

    x2 = torchaudio.functional.resample(
        waveform=x2,
        orig_freq=sr2,
        new_freq=22050,
    )

    diff = x2.shape[-1] - x1.shape[-1]
    x1 = pad(x1, pad=(0, diff), mode='constant', value=0)

    pitch1, _ = compute_pitch(audio=x1)
    pitch2, _ = compute_pitch(audio=x2)

    return (pitch1 - pitch2).abs().mean().item()

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    dataset = ExtendDataset(data_dir=args.data_dir)
    mcd_cal = pymcd.Calculate_MCD('dtw_sl')

    logger.add(
        f"{args.flip_wav_dir}/logs/pitch_mcd.log",
        rotation='200 MB'
    )
    
    avg_pitch_mae = 0.0
    average_mcd = 0.0
    for i, utt in enumerate(dataset):
        key = utt['key']

        synthesized_wav_path = f"{args.flip_wav_dir}/{key}_generated_e2e.wav"
        source_wav_path = dataset.key2wav[key]

        pitch_mae = compute_pitch_mae(
            source_wav_path,
            synthesized_wav_path,
        )

        avg_pitch_mae += (pitch_mae - avg_pitch_mae) / (i + 1)

        mcd = mcd_cal.calculate_mcd(source_wav_path, synthesized_wav_path)
        average_mcd += (mcd - average_mcd) / (i + 1)

        logger.info(
            f"{key}: pitch mae is {pitch_mae}, mcd is {mcd}"
        )

    logger.info(f"The frame-level average pitch mae is {avg_pitch_mae}")
    logger.info(f"The Averaged MCD score is {average_mcd}")