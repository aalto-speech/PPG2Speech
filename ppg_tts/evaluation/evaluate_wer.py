import torch
from ..utils import build_parser, remove_punc_and_tolower
from ..dataset import PersoDatasetBasic
from ..models import PPGFromWav2Vec2Pretrained
from loguru import logger
from torchmetrics.functional.text import word_error_rate

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    dataset = PersoDatasetBasic(args.data_dir, 16000)
    ASRModel = PPGFromWav2Vec2Pretrained(args.asr_pretrained)

    logger.info(f"Evaluating WER on {args.data_dir}, in total {len(dataset)} utterances.")

    ref = []
    pred = []

    for i, utterance in enumerate(dataset):
        wav = utterance["feature"]
        text = remove_punc_and_tolower(utterance['text'])
        ref.append(text)
        ppg = ASRModel.forward(wav)

        predicted_ids = torch.argmax(ppg, dim=-1)
        predicted_trans = ASRModel.processor.batch_decode(predicted_ids)[0].lower()

        pred.append(predicted_trans)

        logger.info(f"{utterance['key']}:\nref: {text}\nhyp: {predicted_trans}\n")
    
    wer = word_error_rate(pred, ref)

    logger.info(f"Evaluating {len(dataset)} utterances, Word Error Rate {wer.item()}")