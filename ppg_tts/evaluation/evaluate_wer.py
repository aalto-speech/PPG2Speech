import torch
import torchaudio
import os
from ..utils import build_parser, remove_punc_and_tolower
from loguru import logger
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from torchmetrics.functional.text import word_error_rate, char_error_rate

def read_wav_scp_text(scp: str, text: str):
    with open(scp, "r") as reader:
        lines = reader.readlines()

    d = {line.strip(' \n').split()[0]: line.strip(' \n').split()[1] for line in lines}

    with open(text, "r") as reader:
        lines = reader.readlines()

    key2text = {}

    for line in lines:
        key, *text = line.split(' ')

        text = remove_punc_and_tolower(" ".join(text).strip("\n"))

        key2text[key] = text

    for k, v in d.items():
        x, sr = torchaudio.load(v)
        x = torchaudio.functional.resample(x, orig_freq=sr, new_freq=16000)
        yield x, key2text[k], k

def read_wav_text(wav_dir: str, text_file: str):
    wavs = os.listdir(wav_dir)

    wavs = [os.path.join(wav_dir, wav) for wav in wavs if "generated_e2e" in wav]

    with open(text_file, "r") as reader:
        lines = reader.readlines()

    key2text = {}

    for line in lines:
        key, *text = line.split(' ')

        text = remove_punc_and_tolower(" ".join(text).strip("\n"))

        key2text[key] = text

    for wav in wavs:
        key = wav.split('/')[-1].replace('_generated_e2e.wav', '')
        x, sr = torchaudio.load(wav)

        x = torchaudio.functional.resample(x, orig_freq=sr, new_freq=16000)

        yield x, key2text[key], key

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    processor = Wav2Vec2Processor.from_pretrained(args.asr_pretrained)
    model = Wav2Vec2ForCTC.from_pretrained(args.asr_pretrained)
    model = model.to(device)

    logger.add(
        f"{args.flip_wav_dir}/logs/wer_cer.log",
        rotation='200 MB'
    )

    logger.info(f"Evaluating WER on {args.flip_wav_dir}, transcripts in {args.data_dir}/text.")

    ref = []
    pred = []

    with torch.inference_mode():
        # for utterance in read_wav_scp_text('/scratch/work/liz32/ppg_tts/data/spk_sanity/wav.scp',
        #                                    '/scratch/work/liz32/ppg_tts/data/spk_sanity/text'):
        for utterance in read_wav_text(args.flip_wav_dir, f"{args.data_dir}/text"):
            wav = utterance[0].to(device)
            text = utterance[1]
            ref.append(text)
            logits = model.forward(wav).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            predicted_trans = processor.batch_decode(predicted_ids)[0].lower()

            pred.append(predicted_trans)

            logger.info(f"{utterance[-1]}:\nref: {text}\nhyp: {predicted_trans}\n")
    
    wer = word_error_rate(pred, ref)

    cer = char_error_rate(pred, ref)

    logger.info(f"Evaluating utterances in {args.flip_wav_dir}, WER {wer.item()}, CER {cer.item()}")