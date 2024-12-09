import torchaudio
from ..utils import build_parser
from ..dataset import PersoDatasetBasic, BaseDataset
from ..models import PPGFromWav2Vec2Pretrained
from loguru import logger
from kaldiio import WriteHelper


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.dataset == 'perso':
        dataset = PersoDatasetBasic(args.data_dir, 16000)
    else:
        dataset = BaseDataset(args.data_dir, 16000)
    
    ASRModel = PPGFromWav2Vec2Pretrained(args.asr_pretrained, no_ctc=args.no_ctc)

    logger.info(f"Extracting PPG to {args.data_dir}, in total {len(dataset)} utterances.")

    flag = "_no_ctc" if args.no_ctc else ""
    with WriteHelper(f"ark,scp,f:{args.data_dir}/ppg{flag}.ark,{args.data_dir}/ppg{flag}.scp") as writer:
        try:
            for i, utterance in enumerate(dataset):
                if args.dataset == 'perso':
                    wav = utterance["feature"]
                else:
                    wav = utterance[1]
                ppg = ASRModel.forward(wav)

                ppg = ppg.squeeze(0)
                if args.dataset == 'perso':
                    key = utterance["key"]
                else:
                    key = utterance[0]
                writer(key, ppg.numpy())
                logger.info(f"{key}: wav length {wav.size(-1)}, ppg shape {ppg.shape}")
        except IndexError:
            logger.info(f"PPG extraction finished")
        except Exception as e:
            logger.error(f"{e}")
    