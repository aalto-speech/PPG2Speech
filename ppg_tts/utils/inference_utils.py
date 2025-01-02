import torch
import json
from scipy.io.wavfile import write
from pathlib import Path
from typing import Tuple
from ..models import VQVAEMatcha
from vocoder.hifigan.env import AttrDict
from vocoder.hifigan.models import Generator as HiFiGAN

def load_VQVAEMatcha(ckpt_path: str, device: str='cpu') -> Tuple[torch.nn.Module, int, float]:
    ckpt = torch.load(ckpt_path, map_location=device)

    with open(ckpt['hyper_parameters']['pitch_stats'], "r") as reader:
            pitch_stats = json.load(reader)

    model = VQVAEMatcha(
        **ckpt['hyper_parameters'],
        **pitch_stats,
    )

    weights = {k.replace('model.', ''): v for k, v in ckpt['state_dict'].items()}

    model.load_state_dict(weights)

    model = model.eval()

    return model, ckpt['hyper_parameters']['diff_steps'], ckpt['hyper_parameters']['temperature']

def load_hifigan(checkpoint_path: str | Path, device: str='cpu'):
    json_config = Path(checkpoint_path).parent / 'config.json'
    with open(json_config, 'r') as reader:
        v1 = json.load(reader)
    h = AttrDict(v1)
    hifigan = HiFiGAN(h).to(device)
    hifigan.load_state_dict(torch.load(checkpoint_path, map_location=device)["generator"])
    _ = hifigan.eval()
    hifigan.remove_weight_norm()
    return hifigan

def mask_to_length(mask: torch.Tensor) -> torch.Tensor:
     """
     converting a mask of shape (B, Tmax) to (B,)
     The masked location is True
     """
     lengths = (~mask).sum(dim=1)
     return lengths

def make_single_audio_mask(w2v2_len: int, pitch_len: int):
    w2v2_mask = torch.full((1, w2v2_len), False)
    pitch_mask = torch.full((1, pitch_len), False)

    return w2v2_mask, pitch_mask

def write_wav(output_dir: str, wav_name: str, wav: torch.Tensor, wav_length: torch.Tensor):
    path = f"{output_dir}/{wav_name}"

    audio = wav[:wav_length]

    audio = audio * 32768.0
    audio = audio.cpu().numpy().astype('int16')

    write(path, 22050, audio)
    print(path, flush=True)
