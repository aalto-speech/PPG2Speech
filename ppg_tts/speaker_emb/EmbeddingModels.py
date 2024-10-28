from pyannote.audio import Model
from pyannote.audio import Inference
import torch

class SpeakerEmbeddingPretrained:
    def __init__(self, auth_token: str):
        self.model = Model.from_pretrained("pyannote/embedding", use_auth_token=auth_token)
        self.inference = Inference(self.model, window="whole")

    def forward(self, waveform_in_memory: dict) -> torch.Tensor:
        return self.inference(waveform_in_memory)
    
