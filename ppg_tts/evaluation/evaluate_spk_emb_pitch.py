import torchaudio
import shutil
import wespeaker
import penn
from functools import partial
from loguru import logger
from scipy.stats import pearsonr
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

def compute_pitch_mae(audio1: str, audio2: str) -> float:
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

    pitch1, _ = compute_pitch(audio=x1)
    pitch2, _ = compute_pitch(audio=x2)

    min_length = min(pitch1.size(-1), pitch2.size(-1))

    pitch1 = pitch1[:, :min_length]
    pitch2 = pitch2[:, :min_length]

    pitch1 = (pitch1 - pitch1.mean()) / pitch1.var()

    pitch2 = (pitch2 - pitch2.mean()) / pitch2.var()

    return (pitch1 - pitch2).abs().mean().item()


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    dataset = ExtendDataset(data_dir=args.data_dir)
    SpEmModel = wespeaker.load_model('vblinkf')

    with open(f"{args.flip_wav_dir}/speaker_mapping", "r") as reader:
        mapping = reader.readlines()
    
    avg_simi = 0.0
    avg_pitch_mae = 0.0
    for entry in mapping:
        source_key, target_key = entry.strip("\n").split()

        synthesized_wav_path = f"{args.flip_wav_dir}/{source_key}_generated_e2e.wav"
        source_wav_path = dataset.key2wav[source_key]
        target_wav_path = dataset.key2wav[target_key]
        target_idx = dataset.key2idx[target_key]

        similarity = SpEmModel.compute_similarity(
            synthesized_wav_path,
            target_wav_path
        )

        pitch_mae = compute_pitch_mae(
            source_wav_path,
            synthesized_wav_path,
        )

        avg_simi += similarity

        avg_pitch_mae += pitch_mae

        logger.info(
            f"Cosine similarity between {source_key} and {target_key} is {similarity},"
            f"The pitch mae is {pitch_mae}"
        )

        if args.debug:
            shutil.copyfile(target_wav_path, f"{args.flip_wav_dir}/{source_key}_speaker_reference.wav")

            shutil.copyfile(source_wav_path, f"{args.flip_wav_dir}/{source_key}_context_reference.wav")

    avg_simi /= len(dataset)
    avg_pitch_mae /= len(dataset)

    logger.info(f"The average cosine similarity is {avg_simi}, average pitch mae is {avg_pitch_mae}")
