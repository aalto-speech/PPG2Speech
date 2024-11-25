import torchaudio
from loguru import logger
from ..utils import build_parser, extract_f0_from_utterance
from ..dataset import PersoDatasetBasic, VCTKBase
from kaldiio import WriteHelper

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.dataset == 'perso':
        dataset = PersoDatasetBasic(args.data_dir)
    elif args.dataset == 'vctk':
        dataset = VCTKBase(data_dir=args.data_dir)
    logger.info(f"Extracting log F0 to {args.data_dir}, in total {len(dataset)} utterances.")

    with WriteHelper(f"ark,scp,f:{args.data_dir}/log_f0.ark,{args.data_dir}/log_f0.scp") as writer:
        with WriteHelper(f"ark,scp,f:{args.data_dir}/voiced.ark,{args.data_dir}/voiced.scp") as voiced_writer:
            for d in dataset:
                if args.dataset == 'perso':
                    key, f0, v_flag = extract_f0_from_utterance(d)
                elif args.datset == 'vctk':
                    key = d[0]
                    _, f0, v_flag = extract_f0_from_utterance({'feature': d[1],
                                                               'key': key})
                writer(key, f0)
                voiced_writer(key, v_flag)