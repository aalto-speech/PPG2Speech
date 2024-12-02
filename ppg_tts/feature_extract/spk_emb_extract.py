from ..utils import build_parser
from ..dataset import PersoDatasetBasic, VCTKLibriTTSRBase
from ..models import SpeakerEmbeddingPretrained
from loguru import logger
from kaldiio import WriteHelper

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.dataset == 'perso':
        dataset = PersoDatasetBasic(args.data_dir)
    elif args.dataset == 'vctk' or args.dataset == 'librittsr':
        dataset = VCTKLibriTTSRBase(data_dir=args.data_dir)
    else:
        raise ValueError(f"Currently dataset {args.dataset} is not supported")

    SpEmModel = SpeakerEmbeddingPretrained(args.auth_token, args.device)

    logger.info(f"Extracting Speaker Embedding to {args.data_dir}, in total {len(dataset)} utterances.")

    with WriteHelper(f"ark,scp,f:{args.data_dir}/embedding.ark,{args.data_dir}/embedding.scp") as writer:
        try:
            for i, utterance in enumerate(dataset):
                if args.dataset == 'perso':
                    wav = utterance["feature"]
                elif args.dataset == 'vctk' or args.dataset == 'librittsr':
                    wav = utterance[1]
                emb = SpEmModel.forward(wav)

                if args.dataset == 'perso':
                    key = utterance["key"]
                elif args.dataset == 'vctk' or args.dataset == 'librittsr':
                    key = utterance[0]
                writer(key, emb)
                logger.info(f"{key}: wav length {wav.size(-1)}, emb shape {emb.shape}")
        except IndexError:
            logger.info(f"Speaker Embedding extraction finished")
        except Exception as e:
            logger.error(f"{e}")
