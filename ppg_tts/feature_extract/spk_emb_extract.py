from ..utils import build_parser
from ..dataset import PersoDatasetBasic
from ..models import SpeakerEmbeddingPretrained
from loguru import logger
from kaldiio import WriteHelper

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    dataset = PersoDatasetBasic(args.data_dir)
    SpEmModel = SpeakerEmbeddingPretrained(args.auth_token, args.device)

    logger.info(f"Extracting Speaker Embedding to {args.data_dir}, in total {len(dataset)} utterances.")

    with WriteHelper(f"ark,scp:{args.data_dir}/embedding.ark,{args.data_dir}/embedding.scp") as writer:
        for i, utterance in enumerate(dataset):
            wav = utterance["feature"]
            emb = SpEmModel.forward(wav)
            writer(utterance['key'], emb)
            logger.info(f"{utterance['key']}: wav length {wav.size(-1)}, emb shape {emb.shape}")
