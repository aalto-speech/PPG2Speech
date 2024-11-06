from ..utils import build_parser
from ..dataset import PersoDatasetBasic
from ..models import PPGFromWav2Vec2Pretrained, PPGFromWav2Vec2PretrainedNoCTC
from loguru import logger
from kaldiio import WriteHelper

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    dataset = PersoDatasetBasic(args.data_dir, 16000)
    if not args.no_ctc:
        ASRModel = PPGFromWav2Vec2Pretrained(args.asr_pretrained)
    else:
        ASRModel = PPGFromWav2Vec2PretrainedNoCTC(args.asr_pretrained)

    logger.info(f"Extracting PPG to {args.data_dir}, in total {len(dataset)} utterances.")

    flag = "_no_ctc" if args.no_ctc else ""
    with WriteHelper(f"ark,scp:{args.data_dir}/ppg{flag}.ark,{args.data_dir}/ppg{flag}.scp") as writer:
        for i, utterance in enumerate(dataset):
            wav = utterance["feature"]
            ppg = ASRModel.forward(wav)

            ppg = ppg.squeeze(0)
            writer(utterance["key"], ppg.numpy())
            logger.info(f"{utterance['key']}: wav length {wav.size(-1)}, ppg shape {ppg.shape}")
    