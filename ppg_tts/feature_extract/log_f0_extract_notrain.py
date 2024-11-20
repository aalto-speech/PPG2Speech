from loguru import logger
from ..utils import build_parser, extract_f0_from_utterance
from ..dataset import PersoDatasetBasic
from kaldiio import WriteHelper

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    dataset = PersoDatasetBasic(args.data_dir)

    logger.info(f"Extracting log F0 to {args.data_dir}, in total {len(dataset)} utterances.")

    with WriteHelper(f"ark,scp,f:{args.data_dir}/log_f0.ark,{args.data_dir}/log_f0.scp") as writer:
        with WriteHelper(f"ark,scp,f:{args.data_dir}/voiced.ark,{args.data_dir}/voiced.scp") as voiced_writer:
            for d in dataset:
                key, f0, v_flag = extract_f0_from_utterance(d)
                writer(key, f0)
                voiced_writer(key, v_flag)