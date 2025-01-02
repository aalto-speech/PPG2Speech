import argparse
import random
import numpy as np
import torch
from loguru import logger
from pathlib import Path
from lightning import LightningModule

from ..utils import load_VQVAEMatcha, load_hifigan, mask_to_length

DEFAULT_OPSET = 15

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class VQVAEMatchaWithVocoder(LightningModule):
    def __init__(self, MatchaVC, vocoder):
        super().__init__()

        self.MatchaVC = MatchaVC
        self.vocoder = vocoder

    def forward(self,
                w2v2_hid: torch.Tensor,
                w2v2_hid_mask: torch.Tensor,
                spk_emb: torch.Tensor,
                f0: torch.Tensor,
                periodicity: torch.Tensor,
                f0_mask: torch.Tensor,
                scales: torch.Tensor):
        diffusion_steps = int(scales[0])
        temperature = scales[1]

        mel, _, _, _ = self.MatchaVC.synthesis(
            x=w2v2_hid,
            x_mask=w2v2_hid_mask,
            spk_emb=spk_emb,
            pitch_target=f0,
            v_flag=periodicity,
            mel_mask=f0_mask,
            diff_steps=diffusion_steps,
            temperature=temperature,
        )

        mel = mel.transpose(-1, -2)

        wav = self.vocoder(mel).clamp(-1, 1)

        wav_length = mask_to_length(f0_mask) * 256

        return wav.squeeze(1), wav_length
    
def get_inputs():
    w2v2_length = 50
    pitch_length = 86

    w2v2 = torch.randn((1, w2v2_length, 1024))
    w2v2_mask = torch.ones((1, w2v2_length)) < 0
    spk_emb = torch.randn((1, 256))
    f0 = torch.randn((1, pitch_length))
    periodicity = torch.randn((1, pitch_length))
    f0_mask = torch.ones((1, pitch_length)) < 0

    scales = torch.tensor([10, 0.667])

    inputs = [w2v2, w2v2_mask, spk_emb, f0, periodicity, f0_mask, scales]
    input_names = ['w2v2_hid', 'w2v2_hid_mask', 'spk_emb', 'f0', 'periodicity', 'f0_mask', 'scales']

    return tuple(inputs), input_names

def main():
    parser = argparse.ArgumentParser('Export VQVAEMatcha to ONNX')

    parser.add_argument(
        '--ckpt_path',
        type=str,
        help='checkpoint path of the VQVAEMatcha model'
    )

    parser.add_argument(
        '--vocoder_ckpt',
        type=str,
        help='checkpoint path of HiFiGAN Vocoder'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        help='output directory of the onnx model'
    )

    parser.add_argument("--opset", type=int, default=15, help="ONNX opset version to use (default 15")

    args = parser.parse_args()

    logger.info(f"Loading VQVAEMatcha from {args.ckpt_path}")

    model, _, _ = load_VQVAEMatcha(args.ckpt_path)

    logger.info(f"Loading HiFiGAN vocoder from {args.vocoder_ckpt}")

    vocoder = load_hifigan(args.vocoder_ckpt)

    fused_model = VQVAEMatchaWithVocoder(model, vocoder)

    output_names = ['wav', 'wav_lengths']

    dummy_input, input_names = get_inputs()

    dynamic_axes = {
        'w2v2_hid': {0: 'batch_size', 1: 'time'},
        'w2v2_hid_mask': {0: 'batch_size', 1: 'time'},
        'spk_emb': {0: 'batch_size'},
        'f0': {0: 'batch_size', 1: 'time'},
        'periodicity': {0: 'batch_size', 1: 'time'},
        'f0_mask': {0: 'batch_size', 1: 'time'},
        'wav': {0: "batch_size", 1: "time"},
        "wav_lengths": {0: "batch_size"},
    }

    Path(args.output_dir).parent.mkdir(parents=True, exist_ok=True)

    fused_model.to_onnx(
        args.output_dir,
        dummy_input,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        export_params=True,
        do_constant_folding=True,
    )

    logger.info(f"ONNX model export to {args.output_dir}")

if __name__ == '__main__':
    main()
