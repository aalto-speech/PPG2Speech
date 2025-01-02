import argparse
import torchaudio
import penn
import random
import numpy as np
from time import perf_counter
from transformers import Wav2Vec2ForCTC
import torch
from loguru import logger
from pathlib import Path
from ..utils import load_VQVAEMatcha, load_hifigan, mask_to_length, make_single_audio_mask, write_wav

SEED = 17
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def inference_single_audio(audio_path: str,
                           spk_emb: torch.Tensor,
                           temperature: float,
                           diff_steps: int,
                           w2v2_model: Wav2Vec2ForCTC,
                           vqvae_matcha: torch.nn.Module,
                           vocoder: torch.nn.Module,
                           target_pitch_median: float,
                           ):
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

    periodicity = torch.log(periodicity)
    pitch = torch.log(pitch)
    pitch_median = pitch.median()
    pitch = pitch * target_pitch_median / pitch_median

    penn_infer = perf_counter() - t0 - w2v2_infer

    w2v2_mask, pitch_mask = make_single_audio_mask(w2v2_hidden.shape[1], pitch.shape[1])

    with torch.no_grad():
        mel, _, _, _ = vqvae_matcha.synthesis(
            x=w2v2_hidden,
            x_mask=w2v2_mask,
            spk_emb=spk_emb,
            pitch_target=pitch,
            v_flag=periodicity,
            mel_mask=pitch_mask,
            diff_steps=diff_steps,
            temperature=temperature,
        )

        vc_infer = perf_counter() - t0 - w2v2_infer - penn_infer

        mel = mel.transpose(-1, -2)

        wav = vocoder(mel).clamp(-1, 1)

        vocoder_infer = perf_counter() - t0 - w2v2_infer - penn_infer - vc_infer

    wav_length = mask_to_length(pitch_mask) * 256

    wav_secs = wav_length / 22050

    logger.info(f"W2V2 inference: {w2v2_infer:.3f} seconds")
    logger.info(f"Penn inference: {penn_infer:.3f} seconds")
    logger.info(f"VQVAEMatcha E2E inference: {vc_infer:.3f} seconds")
    logger.info(f"Vocoder inference: {vocoder_infer:.3f} seconds")
    logger.info(f"Overall RTF: {(vc_infer + w2v2_infer + penn_infer + vocoder_infer) / wav_secs.item()}")

    return wav.squeeze(), wav_length


def main():
    parser = argparse.ArgumentParser('E2E Voice Conversion using W2V2 hidden.')

    parser.add_argument(
        '--model',
        type=str,
        help='checkpoint path of the VQVAEMatcha model'
    )

    parser.add_argument(
        '--vocoder_ckpt',
        type=str,
        help='checkpoint path of HiFiGAN Vocoder'
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

    parser.add_argument(
        "--diff_steps",
        type=int,
        default=10,
        help="The number of steps in reverse diffusion",
    )

    parser.add_argument(
        "--target_pitch_median",
        type=float,
        help="The pitch median of the target speaker "
    )

    args = parser.parse_args()

    logger.info(f"Loading VQVAEMatcha from {args.model}")

    model, _, _ = load_VQVAEMatcha(args.model)

    logger.info(f"Loading HiFiGAN vocoder from {args.vocoder_ckpt}")

    vocoder = load_hifigan(args.vocoder_ckpt)

    logger.info(f"Load W2V2 from GetmanY1/wav2vec2-large-fi-150k-finetuned")
    w2v2_model = Wav2Vec2ForCTC.from_pretrained("GetmanY1/wav2vec2-large-fi-150k-finetuned")

    logger.info(f"Load speaker embedding from {args.speaker_emb}")
    spk_emb = torch.tensor(np.load(args.speaker_emb)[np.newaxis])

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    wav, wav_length = inference_single_audio(
        audio_path=args.audio,
        spk_emb=spk_emb,
        temperature=args.temperature,
        diff_steps=args.diff_steps,
        w2v2_model=w2v2_model,
        vqvae_matcha=model,
        vocoder=vocoder,
        target_pitch_median=args.target_pitch_median,
    )

    wav_name = Path(args.audio).stem + "_generated_e2e.wav"

    write_wav(
        output_dir=args.output_dir,
        wav_name=wav_name,
        wav=wav,
        wav_length=wav_length,
    )

if __name__ == '__main__':
    main()
