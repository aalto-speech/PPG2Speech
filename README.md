# PPG-based Speech Synthesis and Editing for Finnish

## Data
Kaldi-style dataset
1. wav.scp
2. ppg.scp & ppg.ark
3. embedding.scp & embedding.ark
4. log_f0.scp & log_f0.ark (From PENN)
5. voiced.scp & voiced.ark (Periodicity from PENN).
6. Optionally text

## Inference

## Evaluation

### Copy-synthesis Evaluation
See `ppg_tts/evaluation/evaluate_copy_synthesis.sh`

### Cross-speaker Evaluation
See `ppg_tts/evaluation/evaluate_switch_speaker.sh`

### PPG Evaluation for Synthesized Speech (WIP)

#### 1. Synthesize with 1 character's prob switch to another character
First, use alignment to select editing region:
- [x] Use dtw to find alignment between ppg and text
- [x] random select a character in the string, move its probability in the alignment section to another randomly select character (eg. cat -> bat)
- [x] Return edited text and PPG

#### 2. Synthesize edited text with Matcha-TTS baseline
- [x] Move the ONNX inference code here and use ONNX runtime

#### 3. Synthesize edited PPG with the model

#### 4. Evaluate Kaldi PPG/pdf-post

Extract PPG/pdf-post for TTS speech and PPG-synthesized speech.
Evaluate frame-level [Jennsen-Shannon Divergence](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html) and [Wasserstein distance](https://docs.scipy.org/doc/scipy-1.15.2/reference/generated/scipy.stats.wasserstein_distance_nd.html) in the editied region.


## Training
1. Prepare data as the [Data]() section
2. Modify `config/data_template.yaml` and `config/fit_ppgmatcha.yaml` accordingly, or prepare your own config file.
3. Run the following code:
```
python -m ppg_tts.main -c config/data_template.yaml -c config/fit_ppgmatcha.yaml
```

## Reference
