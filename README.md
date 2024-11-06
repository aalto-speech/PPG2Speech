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

### 2024.10.27
Argparser, PPG and Speaker Embedding module done. Needs tests.
perso_data processing scripts done.

### 2024.10.30
Add F0 and energy to features. Perso dataset with conditions finished.

Refactor code.

**Discuss with Lauri on the wav2vec2 ASR subsampling issue. The PPGs don't have the same length as waveform. Maybe need to train a upsampling encoder to reverse this, or learning length regulator to expand PPG length to Mel length.**

## 2024.10.31

Discussed with Lauri, use `torch interpolate` to deal with different sample rate between PPG and Mel-spectromgram.

Finished some modules in the system.Calculate F0_min, F0_max, energy_min, enery_max from the dataset.

Variation adapter need testing.

Ready to build Conformer TTS model.

## 2024.11.01

Training, Validation, Testing loop done. Can proceed to training.

## 2024.11.06

- [ ] Use w2v2 representations to train a model.
- [x] Plot generated mel
- [ ] set up vocoder (starts with universal w/o fine-tuning)
