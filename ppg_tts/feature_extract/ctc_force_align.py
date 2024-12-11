import ctc_segmentation
import torchaudio
import numpy as np
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

    vocab = ASRModel.processor.tokenizer.get_vocab()

    inv_vocab = {v:k for k,v in vocab.items()}

    char_list = [inv_vocab[i] for i in range(len(inv_vocab))]

    config = ctc_segmentation.CtcSegmentationParameters(char_list=char_list)

    config.index_duration = 1 / 50

    with WriteHelper(f"ark,t,f:{args.data_dir}/force_align.ark") as writer:
        for i, utterance in enumerate(dataset):
            if args.dataset == 'perso':
                wav = utterance["feature"]
                text = utterance['text']
            else:
                wav = utterance[1]
                text = utterance[-1]
            ppg = ASRModel.forward(wav)

            text = ASRModel.processor(text=text, return_tensors='np')['input_ids']

            text = np.squeeze(text, axis=0)

            ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_token_list(config, text=[text])

            lst = []
            for chars in ground_truth_mat:
                for char in chars:
                    lst.append(char)
            
            print(lst)

            timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(config, ppg.squeeze().numpy(), ground_truth_mat)

            print(timings)
    
            # if args.dataset == 'perso':
            #     key = utterance["key"]
            # else:
            #     key = utterance[0]
            # writer(key, align[0].numpy())
    