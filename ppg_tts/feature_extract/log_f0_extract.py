import librosa
from loguru import logger
from ..utils import build_parser, extract_f0_from_utterance
from ..dataset import PersoDatasetBasic
from kaldiio import WriteHelper

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    # dataset = PersoDatasetBasic(args.data_dir)

    jid = int(args.jobid)

    with open(f"{args.data_dir}/wav_{jid:02}.scp", "r") as scp_reader:
        utts = scp_reader.readlines() 

    logger.info(f"Extracting log F0 to {args.data_dir}/log_f0_{jid:02}.scp, in total {len(utts)} utterances.")

    with WriteHelper(f"ark,scp,f:{args.data_dir}/log_f0_{jid:02}.ark,{args.data_dir}/log_f0_{jid:02}.scp") as writer:
        with WriteHelper(f"ark,scp,f:{args.data_dir}/voiced_{jid:02}.ark,{args.data_dir}/voiced_{jid:02}.scp") as voiced_writer:
            for utt in utts:
                key, path = utt.split(" ")
                path = path.strip("\n")
                wav, _ = librosa.load(path=path)
                key, f0, v_flag = extract_f0_from_utterance({'key': key, 'feature': wav})
                writer(key, f0)
                voiced_writer(key, v_flag)