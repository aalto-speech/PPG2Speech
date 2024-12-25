import torch
import torchaudio
import os
from ..utils import build_parser, remove_punc_and_tolower
from ..models import PPGFromWav2Vec2Pretrained
from loguru import logger
from torchmetrics.functional.text import word_error_rate, char_error_rate

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

    ASRModel = PPGFromWav2Vec2Pretrained(args.asr_pretrained)

    logger.info(f"Evaluating WER on {args.flip_wav_dir}, transcripts in {args.data_dir}/text.")

    ref = []
    pred = []

    for utterance in read_wav_text(args.flip_wav_dir, f"{args.data_dir}/text"):
        wav = utterance[0]
        text = utterance[1]
        ref.append(text)
        ppg = ASRModel.forward(wav)

        predicted_ids = torch.argmax(ppg, dim=-1)
        predicted_trans = ASRModel.processor.batch_decode(predicted_ids)[0].lower()

        pred.append(predicted_trans)

        logger.info(f"{utterance[-1]}:\nref: {text}\nhyp: {predicted_trans}\n")
    
    wer = word_error_rate(pred, ref)

    cer = char_error_rate(pred, ref)

    logger.info(f"Evaluating utterances in {args.flip_wav_dir}, WER {wer.item()}, CER {cer.item()}")