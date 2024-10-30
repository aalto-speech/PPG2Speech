from ..utils import build_parser
from ..dataset import PersoDatasetBasic
from ..models import PPGFromWav2Vec2Pretrained
from loguru import logger
from kaldiio import WriteHelper
import torch

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    dataset = PersoDatasetBasic(args.data_dir)
    ASRModel = PPGFromWav2Vec2Pretrained(args.asr_pretrained)

    logger.info(f"Extracting PPG to {args.data_dir}, in total {len(dataset)} utterances.")

    with WriteHelper(f"ark,scp:{args.data_dir}/ppg.ark,{args.data_dir}/ppg.scp") as writer:
        for i, utterance in enumerate(dataset):
            wav = utterance["feature"]
            ppg = ASRModel.forward(wav)

            ppg = ppg.squeeze(0)[:, 4:]

            ppg = ppg / torch.sum(ppg, dim=-1, keepdim=True)
            writer(utterance["key"], ppg.numpy())
            logger.info(f"{utterance['key']}: wav length {wav.size(-1)}, ppg shape {ppg.shape}")
    