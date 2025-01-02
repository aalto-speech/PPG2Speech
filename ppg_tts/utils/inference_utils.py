import torch
import json
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
