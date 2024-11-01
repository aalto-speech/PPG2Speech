import numpy as np
from loguru import logger
from ..utils import build_parser
from ..dataset import PersoDatasetBasic
from librosa import pyin
from kaldiio import WriteHelper

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    dataset = PersoDatasetBasic(args.data_dir)

    logger.info(f"Extracting log F0 to {args.data_dir}, in total {len(dataset)} utterances.")

    with WriteHelper(f"ark,scp:{args.data_dir}/log_f0.ark,{args.data_dir}/log_f0.scp") as writer:
        with WriteHelper(f"ark,scp:{args.data_dir}/voiced.ark,{args.data_dir}/voiced.scp") as voiced_writer:
            for i, utterance in enumerate(dataset):
                wav = utterance["feature"]
                foundamental_freq, voiced_flag, _ = pyin(y=wav.numpy(),
                                                         fmin=125,
                                                         fmax=7600,
                                                         sr=22050,
                                                         hop_length=256,
                                                         frame_length=1024)
                foundamental_freq = np.log(foundamental_freq.squeeze())
                writer(utterance["key"], foundamental_freq)
                voiced_writer(utterance["key"], voiced_flag.squeeze().astype(np.int32))
                logger.info(f"{utterance['key']}: wav length {wav.size(-1)}, log_F0 shape {foundamental_freq.shape}")