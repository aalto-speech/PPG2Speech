import torchaudio
import penn
from kaldiio import WriteHelper
from ..utils import build_parser
from loguru import logger
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    with open(f"{args.data_dir}/wav.scp", "r") as scp_reader:
        utts = scp_reader.readlines()
    
    normalize_flag = "_norm" if args.normalize_pitch else ""

    s = StandardScaler()

    logger.info(f"Extracting log F0 to {args.data_dir}/log_f0_penn{normalize_flag}.scp, in total {len(utts)} utterances.")

    with WriteHelper(f"ark,scp,f:{args.data_dir}/log_f0_penn{normalize_flag}.ark,{args.data_dir}/log_f0_penn{normalize_flag}.scp") as writer:
        with WriteHelper(f"ark,scp,f:{args.data_dir}/periodicity{normalize_flag}.ark,{args.data_dir}/periodicity{normalize_flag}.scp") as voiced_writer:
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
                except:
                    logger.warning(f"{key} is too short, should be removed")
                    continue
                if args.normalize_pitch:
                        f0 = s.fit_transform(
                            f0.log().numpy()
                        )
                else:
                    f0 = f0.log().numpy()
                writer(key, f0.squeeze(0))
                voiced_writer(key, period.log().squeeze(0).numpy())