from loguru import logger
from ..utils import build_parser, extract_f0_from_utterance
from ..dataset import PersoDatasetBasic
from kaldiio import WriteHelper
from multiprocessing import Pool

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    dataset = PersoDatasetBasic(args.data_dir)

    logger.info(f"Extracting log F0 to {args.data_dir}, in total {len(dataset)} utterances.")

    with Pool(args.nj) as pool:
        results = pool.map(extract_f0_from_utterance, dataset)

    with WriteHelper(f"ark,scp:{args.data_dir}/log_f0.ark,{args.data_dir}/log_f0.scp") as writer:
        with WriteHelper(f"ark,scp:{args.data_dir}/voiced.ark,{args.data_dir}/voiced.scp") as voiced_writer:
            for result in results:
                writer(result[0], result[1])
                voiced_writer(result[0], result[2])