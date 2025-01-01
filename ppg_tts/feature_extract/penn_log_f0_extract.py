import torch
import torchaudio
import penn
from kaldiio import WriteHelper
from ..utils import build_parser
from loguru import logger


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    with open(f"{args.data_dir}/wav.scp", "r") as scp_reader:
        utts = scp_reader.readlines() 

    logger.info(f"Extracting log F0 to {args.data_dir}/log_f0.scp, in total {len(utts)} utterances.")

    with WriteHelper(f"ark,scp,f:{args.data_dir}/log_f0_penn.ark,{args.data_dir}/log_f0_penn.scp") as writer:
        with WriteHelper(f"ark,scp,f:{args.data_dir}/periodicity.ark,{args.data_dir}/periodicity.scp") as voiced_writer:
            for utt in utts:
                key, path = utt.strip(" \n").split(" ")
                wav, sr = torchaudio.load(path)
                wav = torchaudio.functional.resample(wav, sr, 22050)
                try:
                    f0, period = penn.from_audio(
                        audio=wav,
                        sample_rate=22050,
                        hopsize=(256 / 22050),
                        fmin=40,
                        fmax=700,
                        center='zero',
                        checkpoint=None,
                        interp_unvoiced_at=0.2,
                    )
                    logger.info(f"{key}: f0 length {f0.shape[-1]}")
                    writer(key, f0.log().squeeze(0).numpy())
                    voiced_writer(key, period.log().squeeze(0).numpy())
                except:
                    logger.warning(f"{key} is too short, should be removed")