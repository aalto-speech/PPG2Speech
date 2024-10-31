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
        for i, utterance in enumerate(dataset):
            wav = utterance["feature"]
            foundamental_freq, _, _ = pyin(y=wav.numpy(),
                                           fmin=125,
                                           fmax=7600,
                                           sr=22050,
                                           fill_na=1e-8,
                                           hop_length=256,
                                           frame_length=1024)
            foundamental_freq = np.log(foundamental_freq.squeeze()[:wav.size(-1)])
            writer(utterance["key"], foundamental_freq)
            logger.info(f"{utterance['key']}: wav length {wav.size(-1)}, log_F0 shape {foundamental_freq.shape}")