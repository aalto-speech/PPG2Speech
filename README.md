# Finnish Zero-shot Text-to-Speech (TTS) with Phonetic PosteriorGram (PPG) and Speaker-Embedding

## Idea

### Starting Point (2024.10.27)
+ One trained ASR model extracts PPG ($ppg_x$) from source utterance $x$.
+ One trained Speaker-Embedding model extracts speaker embedding $emb_x$ from utterance $x$ from speaker $s$.
+ Generate Mel-spectrum using $ppg_x$ and condition on $emb_s$. Regression on Mel-spectrum.
+ A trained vocoder to generate wavform $y$, consistency loss between PPG ($ppg_y$) & speaker-embedding ($emb_y$) of generated speech


## Dev Log

### 2024.10.26:
PPG extract model: https://huggingface.co/GetmanY1/wav2vec2-large-fi-150k-finetuned

Speaker Embedding model: https://huggingface.co/pyannote/embedding

Install [pytorch-lighting](https://lightning.ai/docs/pytorch/stable/#install-lightning) and [pyannote.audio](https://github.com/pyannote/pyannote-audio) for training.