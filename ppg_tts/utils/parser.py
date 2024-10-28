import argparse

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Argument parser for PPG and Speaker Embedding extraction.\
                                      TTS training will use Pytorch-lightning CLI.")
    parser.add_argument("--data_dir", 
                        help="The data directory for Perso Dataset. Should contains wav.scp and text.",
                        default="./data")
    parser.add_argument("--auth_token",
                        help="The authorisation token for using Pyannote Embedding model.",
                        required=True)
    parser.add_argument("--asr_pretrained", 
                        help="Pretrained ASR model for PPG extraction. Should compatible with Huggingface pretrained models.",
                        default="GetmanY1/wav2vec2-large-fi-150k-finetuned")
    return parser