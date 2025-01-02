import argparse
import os
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torchaudio
import penn
from pathlib import Path
from loguru import logger
from time import perf_counter
from transformers import Wav2Vec2ForCTC

def make_single_audio_mask(w2v2_len: int, pitch_len: int):
    w2v2_mask = np.full((1, w2v2_len), False)
    pitch_mask = np.full((1, pitch_len), False)

    return w2v2_mask, pitch_mask

def inference_audio(audio_path: str,
                    spk_emb: np.ndarray,
                    temperature: float,
                    w2v2_model: Wav2Vec2ForCTC,
                    vc_model: ort.InferenceSession):
    audio, sr = torchaudio.load(audio_path)

    audio_w2v2 = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=16000)

    audio_pitch = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=22050)

    t0 = perf_counter()

    with torch.no_grad():
        w2v2_hidden = w2v2_model(audio_w2v2, output_hidden_states=True).hidden_states[-1]

    w2v2_infer = perf_counter() - t0

    pitch, periodicity = penn.from_audio(
        audio=audio_pitch,
        sample_rate=22050,
        hopsize=(256 / 22050),
        fmin=40,
        fmax=700,
        center='zero',
        checkpoint=None,
        interp_unvoiced_at=0.2,
    )

    penn_infer = perf_counter() - t0 - w2v2_infer

    w2v2_mask, pitch_mask = make_single_audio_mask(w2v2_hidden.shape[1], pitch.shape[1])

    if pitch.shape[-1] % 2 == 1:
        pitch = torch.nn.functional.pad(pitch, (0, 1), 'replicate')
        periodicity = torch.nn.functional.pad(periodicity, (0, 1), 'replicate')
        pitch_mask = np.pad(pitch_mask, ((0, 0), (0, 1)), 'constant', constant_values=True)

    inputs = {
        'w2v2_hid': w2v2_hidden.numpy(),
        'w2v2_hid_mask': w2v2_mask,
        'spk_emb': spk_emb,
        'f0': pitch.numpy(),
        'periodicity': np.log(periodicity.numpy()),
        'f0_mask': pitch_mask,
        'scales': np.array([10, temperature], dtype=np.float32)
    }

    logger.error("Here is the bug, the dynamic shape of the time axes is not properly followed by ONNX")
    wav, wav_length = vc_model.run(None, input_feed=inputs)

    vc_infer = perf_counter() - t0 - w2v2_infer - penn_infer

    wav_secs = wav_length / 22050

    logger.info(f"W2V2 inference: {w2v2_infer} seconds")
    logger.info(f"Penn inference: {penn_infer} seconds")
    logger.info(f"VQVAEMatcha E2E inference: {vc_infer} seconds")
    logger.info(f"Overall RTF: {(vc_infer + w2v2_infer + penn_infer) / wav_secs}")

    return wav, wav_length

def write_wav(output_dir: str, wav_name: str, wav: np.ndarray, wav_length: np.ndarray):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = f"{output_dir}/{wav_name}"

    audio = wav[:wav_length]

    sf.write(path, audio, 22050, 'PCM_24')

def main():
    parser = argparse.ArgumentParser(description="Inference with VQVAEMatcha ONNX")

    parser.add_argument(
        '--model',
        help="VQVAEMatcha ONNX model location",
        type=str,
    )

    parser.add_argument(
        '--audio',
        help='The audio file of the source speech',
        type=str,
    )

    parser.add_argument(
        '--speaker_emb',
        help='numpy file of the speaker embedding',
        type=str,
    )

    parser.add_argument(
        '--output_dir',
        help="The output directory for the synthesized audio",
        type=str,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.667,
        help="Variance of the x0 noise (default: 0.667)",
    )


    args = parser.parse_args()

    providers = ["CPUExecutionProvider"]

    options = ort.SessionOptions()
    options.intra_op_num_threads = os.cpu_count() - 1  # Use one less than total cores
    options.inter_op_num_threads = 1
    logger.info(f"Load VQVAEMatcha ONNX from {args.model}")
    vc_model = ort.InferenceSession(args.model, providers=providers, sess_options=options)

    logger.info(f"Load W2V2 from GetmanY1/wav2vec2-large-fi-150k-finetuned")
    w2v2_model = Wav2Vec2ForCTC.from_pretrained("GetmanY1/wav2vec2-large-fi-150k-finetuned")

    spk_emb = np.load(args.speaker_emb)[np.newaxis]

    wav, wav_length = inference_audio(
        audio_path=args.audio,
        spk_emb=spk_emb,
        w2v2_model=w2v2_model,
        vc_model=vc_model,
        temperature=args.temperature,
    )

    wav_name = Path(args.audio).stem + "_generated_e2e_onnx.wav"

    write_wav(args.output_dir, wav_name, wav, wav_length)

if __name__ == '__main__':
    main()
