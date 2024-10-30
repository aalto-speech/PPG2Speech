import librosa
import numpy as np
from ..utils import build_parser
from ..dataset import PersoDatasetBasic
from librosa import pyin
from kaldiio import WriteHelper
from tqdm import tqdm

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    dataset = PersoDatasetBasic(args.data_dir)

    print(f"Extracting log F0 to {args.data_dir}")

    with WriteHelper(f"ark,scp:{args.data_dir}/log_f0.ark,{args.data_dir}/log_f0.scp") as writer:
        for i, utterance in tqdm(enumerate(dataset)):
            wav = utterance["feature"]
            foundamental_freq, v_flag, v_prob = pyin(y=wav.numpy(),
                                           fmin=librosa.note_to_hz('C2'),
                                           fmax=librosa.note_to_hz('C7'),
                                           sr=16000,
                                           fill_na=1e-8,
                                           hop_length=1,
                                           frame_length=1024)
            writer(utterance["key"], np.log(foundamental_freq.squeeze()[:wav.size(-1)]))