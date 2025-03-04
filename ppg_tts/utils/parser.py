import argparse

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Argument parser for PPG and Speaker Embedding extraction.\
                                      TTS training will use Pytorch-lightning CLI.")
    parser.add_argument("--dataset",
                        help="The dataset used")
    parser.add_argument("--data_dir", 
                        help="The data directory for Dataset. Should contains wav.scp and text for Perso. Download location for VCTK.",
                        default="./data")
    parser.add_argument("--auth_token",
                        help="The authorisation token for using Pyannote Embedding model.")
    parser.add_argument("--asr_pretrained", 
                        help="Pretrained ASR model for PPG extraction. Should compatible with Huggingface pretrained models.",
                        default="GetmanY1/wav2vec2-large-fi-150k-finetuned")
    parser.add_argument("--device",
                        help="CPU/CUDA device to run",
                        default="cpu")
    parser.add_argument("--no_ctc",
                        help="Whether use ctc output or w2v2 feature",
                        action="store_true")
    parser.add_argument("--jobid",
                        help="jobid in pitch extraction")
    parser.add_argument("--ckpt",
                        help="ckpt path for the TTS/VC model.")
    parser.add_argument("--flip_wav_dir",
                        help="the wav directory for flipped generation")
    parser.add_argument('--debug',
                        help='whether or not using debug mode.',
                        action="store_true")
    parser.add_argument('--switch_speaker',
                        help='whether to switch the speaker embedding during inference',
                        action='store_true')
    parser.add_argument(
        '--normalize_pitch',
        help='whether to normalize the pitch during pitch extraction',
        action='store_true'
    )
    parser.add_argument(
        '--model_class',
        help='The path to the model class',
        type=str,
    )
    return parser