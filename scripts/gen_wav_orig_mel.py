import os
import torchaudio
import numpy as np
from speechbrain.lobes.models.HifiGAN import mel_spectogram
from torchaudio.transforms import Resample
import sys
sys.path.append("/scratch/work/liz32/ppg_tts")
from ppg_tts.utils.fit_utils import plot_mel

audio_samples = [
    "/scratch/elec/t405-puhe/c/perso_synteesi/male/02m/wav/02m_test_0001.wav",
    "/scratch/elec/t405-puhe/c/perso_synteesi/female/03/wav/03_test_0015.wav",
    "/scratch/elec/t405-puhe/c/perso_synteesi/male/06m/wav/06m_test_0007.wav",
    "/scratch/elec/t405-puhe/c/perso_synteesi/female/08/wav/08_test_0005.wav",
    "/scratch/elec/t405-puhe/c/perso_synteesi/male/11m/wav/11m_test_0013.wav",
    "/scratch/elec/t405-puhe/c/perso_synteesi/female/12/wav/12_test_0011.wav",
]

save_dir = sys.argv[1]

os.makedirs(f"{save_dir}", exist_ok=True)
os.makedirs(f"{save_dir}/mel_samples", exist_ok=True)

resampler = Resample(orig_freq=44100, new_freq=22050)

for sample in audio_samples:
    name = os.path.splitext(os.path.basename(sample))[0]
    x, sr = torchaudio.load(sample)
    x = resampler(x)

    mel = mel_spectogram(sample_rate=22050,
                         n_fft=1024,
                         win_length=1024,
                         hop_length=256,
                         f_min=0,
                         f_max=8000,
                         n_mels=80,
                         normalized=False,
                         compression=True,
                         audio=x,
                         power=1,
                         norm="slaney",
                         mel_scale="slaney")
    
    plot_mel(mel.cpu().numpy(), save_dir, key=f"{name}_orig")