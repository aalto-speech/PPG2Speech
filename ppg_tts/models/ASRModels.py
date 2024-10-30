from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch

class PPGFromWav2Vec2Pretrained:
    def __init__(self, pretrained: str="GetmanY1/wav2vec2-large-fi-150k-finetuned", device: str='cpu'):
        self.pretrained = pretrained
        self.processor = Wav2Vec2Processor.from_pretrained(pretrained)
        self.model = Wav2Vec2ForCTC.from_pretrained(pretrained)

        if device == "cuda:0" and torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = 'cpu'
        
        self.model = self.model.to(self.device)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            waveform = waveform.to(self.device)
            logits = self.model(waveform).logits

        return logits
    
    def tokenize(self, target: str):
        encoding = self.processor(text=target)
        return encoding