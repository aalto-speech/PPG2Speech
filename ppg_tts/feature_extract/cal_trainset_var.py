import json
from loguru import logger
from kaldiio import load_scp
from ..utils import build_parser

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    var = 0.0

    d = load_scp(f"{args.data_dir}/ppg_no_ctc.scp")
    
    for i, key in enumerate(d):
        array = d[key]
        var = (var * i + array.var(axis=1).sum() / array.shape[0]) / (i + 1)
        logger.info(f"{key}: current running variance {var}")

    with open(f"{args.data_dir}/variance.json", "w") as writter:
        json.dump(
            {"variance": var},
            writter,
            indent=4
        )

    