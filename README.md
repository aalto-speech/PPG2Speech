# Pronunciation Editing for Finnish Speech using Phonetic Posteriorgrams

This is the official repo for our paper *Pronunciation Editing for Finnish Speech using Phonetic Posteriorgrams* in Speech Synthesis Workshop 2025 (SSW13) at Leeuwarden, the Netherlands.

## Setup your environment

Create a conda environment via:

```bash
conda env create -f environment.yaml
```

## Data

1. Prepare the data in a Kaldi's `wav.scp` format.
2. Use a pre-trained Kaldi HMM-DNN model to extract PPGs from speech. [Kaldi docs](https://kaldi-asr.org/) are helpful to do that.
3. Extract Speaker Embedding using [Wespeaker](https://github.com/wenet-e2e/wespeaker) cli. Specifically, you should use ```wespeaker --task embedding_kaldi --wav_scp YOUR_WAV.scp --output_file /path/to/embedding```, see [here](https://wenet.org.cn/wespeaker/python_package.html#command-line-usage).
4. Extract Pitch and Periodicity using `ppg_tts/feature_extract/penn_log_f0_extract.py`.

## Pretrained Checkpoint:

Pretrained checkpoint is available [here](https://drive.google.com/file/d/1HKPi04xN3a07fv_KzhdK-MlkRKyVyFmZ/view?usp=sharing).

Pretrained HiFi-GAN generator checkpoint is avaible [here](https://drive.google.com/file/d/1uP6iv9dFvKdlCBPZC5VwxYjg8eNR6Da8/view?usp=sharing). Please put the HiFi-GAN checkpoint under `vocoder/hifigan/ckpt`.

## Training

```python
python -m ppg_tts.main fit -c config/fit_ppgmatcha.yaml -c config/data_template.yaml
```

You can overwrite the arguments via CLI, see [pytorch-lightning docs](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced.html).

## Evaluation

### Copy-synthesis/reconstruct-synthesis Evaluation

See [ppg_tts/evaluation/evaluate_copy_synthesis.sh](https://github.com/aalto-speech/PPG2Speech/blob/main/ppg_tts/evaluation/evaluate_copy_synthesis.sh)

### Cross-speaker Evaluation

See [ppg_tts/evaluation/evaluate_switch_speaker.sh](https://github.com/aalto-speech/PPG2Speech/blob/main/ppg_tts/evaluation/evaluate_switch_speaker.sh)

### Phoneme-level Editing

See [ppg_tts/evaluation/evaluate_editing/evaluate_editing.sh](https://github.com/aalto-speech/PPG2Speech/blob/main/ppg_tts/evaluation/evaluate_editing/evaluate_editing.sh)

## Inference

Currently we don't have specific script for inferencing a pre-trained models with minimum efforts, but inference can be done via executing certain stages in the evaluation script. Specialized inference script will be available in the future.

To do inference, data should be prepared the same as the training data (see [here](https://github.com/aalto-speech/PPG2Speech/tree/main?tab=readme-ov-file#data)).

### Inference for TTS

Follow the comment in [ppg_tts/evaluation/evaluate_switch_speaker.sh](https://github.com/aalto-speech/PPG2Speech/blob/main/ppg_tts/evaluation/evaluate_switch_speaker.sh) and set `start=0` and `end=1` to do TTS inference.

### Inference for Phoneme-level Editing

Follow the comment in [ppg_tts/evaluation/evaluate_editing/evaluate_editing.sh](https://github.com/aalto-speech/PPG2Speech/blob/main/ppg_tts/evaluation/evaluate_editing/evaluate_editing.sh) and set `start=0` and `end=1` to do editing inference.

## Citation

Coming soon.

## License

Our work is shared under [Creative Commons Attribution 4.0 International (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/)
