from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from loguru import logger
import torch

class PPGFromWav2Vec2Pretrained:
    def __init__(self, 
                 pretrained: str="GetmanY1/wav2vec2-large-fi-150k-finetuned",
                 device: str='cpu',
                 no_ctc: bool=False,
                 num_hidden: int=1):
        logger.info(f"Loading pretrained model {pretrained}, running on {device}")
        self.pretrained = pretrained
        self.no_ctc = no_ctc
        self.num_hidden = num_hidden
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
            output = self.model(waveform)
            if self.no_ctc:
                if self.num_hidden > 1:
                    raise NotImplementedError("No implementation for aggregating multiple hidden representations yet.")
                else:
                    logits = output.last_hidden_state
            else:
                logits = output.logits

        return logits
    
    def tokenize(self, target: str):
        encoding = self.processor(text=target)
        return encoding
    