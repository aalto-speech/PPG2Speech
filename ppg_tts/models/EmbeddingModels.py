from pyannote.audio import Model
from pyannote.audio import Inference
from loguru import logger
import torch
import torch.nn.functional as f
import numpy as np

class SpeakerEmbeddingPretrained:
    def __init__(self, auth_token: str, device: str):
        logger.info(f"Using pyannote/embedding model, running on {device}")
        self.model = Model.from_pretrained("pyannote/embedding", use_auth_token=auth_token, strict=False)

        self.inference = Inference(self.model, window="whole")

        self.inference.to(torch.device(device))

    def forward(self, waveform_in_memory: torch.Tensor, sr: int=22050) -> np.ndarray:
        len_pad = sr * 5 - waveform_in_memory.size(-1)
        if len_pad > 0:
            waveform_in_memory = f.pad(waveform_in_memory, [0, len_pad], mode='constant', value=0.0)
        return self.inference({"waveform": waveform_in_memory, "sample_rate": sr})
    
