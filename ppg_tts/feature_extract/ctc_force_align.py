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
    
    ASRModel = PPGFromWav2Vec2Pretrained(args.asr_pretrained, no_ctc=False)

    logger.info(f"Extracting force alignment to {args.data_dir}, in total {len(dataset)} utterances.")

    with WriteHelper(f"ark,t,f:{args.data_dir}/force_align.ark") as writer:
        for i, utterance in enumerate(dataset):
            if args.dataset == 'perso':
                wav = utterance["feature"]
                text = utterance['text']
            else:
                wav = utterance[1]
                text = utterance[-1]
            ppg = ASRModel.forward(wav)

            text = ASRModel.processor(text=text, return_tensors="pt")

            align, prob = torchaudio.functional.forced_align(ppg, targets=text['input_ids'])

            if args.dataset == 'perso':
                key = utterance["key"]
            else:
                key = utterance[0]
            writer(key, align[0].numpy())
    