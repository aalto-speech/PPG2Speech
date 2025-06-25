import argparse
import kaldiio
import os
import warnings
from pathlib import Path
import re
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

def plot_spectrogram_to_numpy(spectrogram, filename):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.title("Synthesised Mel-Spectrogram")
    fig.canvas.draw()
    plt.savefig(filename)

def intersperse(lst, item):
    # Adds blank symbol
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result

_pad = "_"
_letters = "abcdefghijklmnopqrstuvwxyzåäö"
_space = " "

fin_symbols = [_pad] + list(_letters) + [_space]
SPACE_ID = fin_symbols.index(" ")

_fin_symbol_to_id = {s: i for i, s in enumerate(fin_symbols)}
_id_to_fin_symbol = {i: s for i, s in enumerate(fin_symbols)}

def fin_text_to_sequence(text):
    sequence = []

    cleaned_text = re.sub(r'[^\w\s]', '', text).lower()

    for char in cleaned_text:
        symbol_id = _fin_symbol_to_id[char]
        sequence.append(symbol_id)

    return sequence, cleaned_text


def process_text(i: int, text: str, device: torch.device):
    print(f"[{i}] - Input text: {text}")
    x = torch.tensor(
        intersperse(fin_text_to_sequence(text)[0], 0),
        dtype=torch.long,
        device=device,
    )[None]
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)

    return {"x_orig": text, "x": x, "x_lengths": x_lengths, "x_phones": None}

def validate_args(args):
    assert (
        args.file
    ), "File must be provided Matcha-T(ea)TTS need sometext to whisk the waveforms."
    assert args.temperature >= 0, "Sampling temperature cannot be negative"
    assert args.speaking_rate >= 0, "Speaking rate must be greater than 0"
    if not os.path.exists(args.file):
        raise ValueError(f"{args.file} not exists!")
    
    if args.spk and not os.path.exists(args.spk):
        raise ValueError(f"Speaker embedding scp must be provided")
    return args


def write_wavs(model, inputs, output_dir, external_vocoder=None, keys=None):
    if external_vocoder is None:
        print("The provided model has the vocoder embedded in the graph.\nGenerating waveform directly")
        t0 = perf_counter()
        wavs, wav_lengths = model.run(None, inputs)
        infer_secs = perf_counter() - t0
        mel_infer_secs = vocoder_infer_secs = None
    else:
        print("[🍵] Generating mel using Matcha")
        mel_t0 = perf_counter()
        mels, mel_lengths = model.run(None, inputs)
        mel_infer_secs = perf_counter() - mel_t0
        print("Generating waveform from mel using external vocoder")
        vocoder_inputs = {external_vocoder.get_inputs()[0].name: mels}
        vocoder_t0 = perf_counter()
        wavs = external_vocoder.run(None, vocoder_inputs)[0]
        vocoder_infer_secs = perf_counter() - vocoder_t0
        wavs = wavs.squeeze(1)
        wav_lengths = mel_lengths * 256
        infer_secs = mel_infer_secs + vocoder_infer_secs

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, (wav, wav_length) in enumerate(zip(wavs, wav_lengths)):
        if keys is None:
            output_filename = output_dir.joinpath(f"output_{i + 1}.wav")
        else:
            output_filename = output_dir.joinpath(f"{keys[i]}_generated_e2e.wav")
        audio = wav[:wav_length]
        print(f"Writing audio to {output_filename}")
        sf.write(output_filename, audio, 22050, "PCM_24")

    wav_secs = wav_lengths.sum() / 22050
    print(f"Inference seconds: {infer_secs}")
    print(f"Generated wav seconds: {wav_secs}")
    rtf = infer_secs / wav_secs
    if mel_infer_secs is not None:
        mel_rtf = mel_infer_secs / wav_secs
        print(f"Matcha RTF: {mel_rtf}")
    if vocoder_infer_secs is not None:
        vocoder_rtf = vocoder_infer_secs / wav_secs
        print(f"Vocoder RTF: {vocoder_rtf}")
    print(f"Overall RTF: {rtf}")


def write_mels(model, inputs, output_dir):
    t0 = perf_counter()
    mels, mel_lengths = model.run(None, inputs)
    infer_secs = perf_counter() - t0

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, mel in enumerate(mels):
        output_stem = output_dir.joinpath(f"output_{i + 1}")
        plot_spectrogram_to_numpy(mel.squeeze(), output_stem.with_suffix(".png"))
        np.save(output_stem.with_suffix(".numpy"), mel)

    wav_secs = (mel_lengths * 256).sum() / 22050
    print(f"Inference seconds: {infer_secs}")
    print(f"Generated wav seconds: {wav_secs}")
    rtf = infer_secs / wav_secs
    print(f"RTF: {rtf}")


def main():
    parser = argparse.ArgumentParser(
        description=" 🍵 Matcha-TTS: A fast TTS architecture with conditional flow matching"
    )
    parser.add_argument(
        "model",
        type=str,
        help="ONNX model to use",
    )
    parser.add_argument("--vocoder", type=str, default=None, help="Vocoder to use (defaults to None)")
    parser.add_argument("--file", type=str, default=None, help="Text file to synthesize (key-text pair)")
    parser.add_argument("--spk", type=str, default=None, help="path of numpy file of the speaker embedding")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.667,
        help="Variance of the x0 noise (default: 0.667)",
    )
    parser.add_argument(
        "--speaking-rate",
        type=float,
        default=1.0,
        help="change the speaking rate, a higher value means slower speaking rate (default: 1.0)",
    )
    parser.add_argument("--gpu", action="store_true", help="Use CPU for inference (default: use GPU if available)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.getcwd(),
        help="Output folder to save results (default: current dir)",
    )

    args = parser.parse_args()
    args = validate_args(args)

    if args.gpu:
        providers = ["CUDAExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    options = ort.SessionOptions()
    options.intra_op_num_threads = os.cpu_count() - 1  # Use one less than total cores
    options.inter_op_num_threads = 1 
    model = ort.InferenceSession(args.model, providers=providers, sess_options=options)

    model_inputs = model.get_inputs()
    model_outputs = list(model.get_outputs())

    with open(args.file, encoding="utf-8") as file:
        lines = file.read().splitlines()

    text_lines = {}
    for line in lines:
        key, *text = line.split()
        text_lines[key] = ' '.join(text)

    processed_lines = {key: process_text(i, line, "cpu") for i, (key, line) in enumerate(text_lines.items())}
    keys = [key for key in processed_lines]

    x = [line["x"].squeeze() for _, line in processed_lines.items()]
    # Pad
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
    x = x.detach().cpu().numpy()
    x_lengths = np.array([line["x_lengths"].item() for _, line in processed_lines.items()], dtype=np.int64)
    inputs = {
        "x": x,
        "x_lengths": x_lengths,
        "scales": np.array([args.temperature, args.speaking_rate], dtype=np.float32),
    }
    is_multi_speaker = len(model_inputs) == 4
    if is_multi_speaker:
        if args.spk is None:
            spk_emb = np.random.rand(1, 512)
            warn = "[!] Speaker ID not provided! Generate a random speaker embedding using torch.randn"
            warnings.warn(warn, UserWarning)
            spk_emb = np.repeat(spk_emb, x.shape[0], axis=0)
        else:
            spk_emb_dict = kaldiio.load_scp(args.spk)
            spks = [spk_emb_dict[key] for key in keys]
            spk_emb = np.vstack(spks).astype(np.float32)
        inputs["spks"] = spk_emb

    has_vocoder_embedded = model_outputs[0].name == "wav"
    if has_vocoder_embedded:
        write_wavs(model, inputs, args.output_dir, keys=keys)
    elif args.vocoder:
        external_vocoder = ort.InferenceSession(args.vocoder, providers=providers)
        write_wavs(model, inputs, args.output_dir, external_vocoder=external_vocoder, keys=keys)
    else:
        warn = "[!] A vocoder is not embedded in the graph nor an external vocoder is provided. The mel output will be written as numpy arrays to `*.npy` files in the output directory"
        warnings.warn(warn, UserWarning)
        write_mels(model, inputs, args.output_dir, keys=keys)


if __name__ == "__main__":
    main()
