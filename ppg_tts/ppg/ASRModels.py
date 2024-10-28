from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch

class PPGFromWav2Vec2Pretrained:
    def __init__(self, pretrained: str="GetmanY1/wav2vec2-large-fi-150k-finetuned"):
        self.pretrained = pretrained
        self.processor = Wav2Vec2Processor.from_pretrained(pretrained)
        self.model = Wav2Vec2ForCTC.from_pretrained(pretrained)

    def forward(self, waveform_in_memory) -> torch.Tensor:
        inputs = waveform_in_memory["waveform"]
        logits = self.model(inputs).logits

        return logits