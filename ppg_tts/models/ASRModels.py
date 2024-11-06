from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from loguru import logger
import torch

class PPGFromWav2Vec2Pretrained:
    def __init__(self, pretrained: str="GetmanY1/wav2vec2-large-fi-150k-finetuned", device: str='cpu'):
        logger.info(f"Loading pretrained model {pretrained}, running on {device}")
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
    
class PPGFromWav2Vec2PretrainedNoCTC(PPGFromWav2Vec2Pretrained):
    def __init__(self, pretrained = "GetmanY1/wav2vec2-large-fi-150k-finetuned", device = 'cpu'):
        super().__init__(pretrained, device)

        self.model = self._drop_ctc_layer()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            waveform = waveform.to(self.device)
            logits = self.model(waveform).last_hidden_state

        return logits

    def _drop_ctc_layer(self):
        logger.info("Drop CTC layer from the model.")
        keep_layers = list(self.model.children())[:-2]

        new_model = torch.nn.Sequential(*keep_layers)

        return new_model