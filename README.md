# Pronunciation Editing for Finnish Speech using Phonetic Posteriorgrams

## Data

1. Prepare the data in a Kaldi's `wav.scp` format.
2. Use a pre-trained Kaldi HMM-DNN model to extract PPGs from speech. [Kaldi docs](https://kaldi-asr.org/) are helpful to do that.
3. Extract Speaker Embedding using [Wespeaker](https://github.com/wenet-e2e/wespeaker) cli.
4. Extract Pitch and Periodicity using `ppg_tts/feature_extract/penn_log_f0_extract.py`.

## Training

```python
python -m ppg_tts.main fit -c config/fit_ppgmatcha.yaml -c config/data_template.yaml
```

You can overwrite the arguments via CLI, see [pytorch-lightning docs](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced.html).

## Evaluation

### Copy-synthesis/reconstruct-synthesis Evaluation

See `ppg_tts/evaluation/evaluate_copy_synthesis.sh`

### Cross-speaker Evaluation

See `ppg_tts/evaluation/evaluate_switch_speaker.sh`

### Editing

See `ppg_tts/evaluation/evaluate_editing/evaluate_editing.sh`

## Reference
