import shutil
import os
import torchaudio
import numpy as np
from speechbrain.lobes.models.HifiGAN import mel_spectogram
from torchaudio.transforms import Resample

audio_samples = [
    "/scratch/elec/t405-puhe/c/perso_synteesi/male/02m/wav/02m_test_0001.wav",
    "/scratch/elec/t405-puhe/c/perso_synteesi/male/07m/wav/07m_test_0011.wav",
    "/scratch/elec/t405-puhe/c/perso_synteesi/male/13m/wav/13m_test_0016.wav",
    "/scratch/elec/t405-puhe/c/perso_synteesi/female/29/wav/29_test_0008.wav",
    "/scratch/elec/t405-puhe/c/perso_synteesi/female/14/wav/14_test_0008.wav",
    "/scratch/elec/t405-puhe/c/perso_synteesi/female/05/wav/05_test_0007.wav",
]

save_dir = "/scratch/work/liz32/ppg_tts/inspect_vocoder"

os.makedirs(f"{save_dir}", exist_ok=True)
os.makedirs(f"{save_dir}/mel", exist_ok=True)
os.makedirs(f"{save_dir}/audio", exist_ok=True)

resampler = Resample(orig_freq=44100, new_freq=22050)

for sample in audio_samples:
    name = os.path.splitext(os.path.basename(sample))[0]
    shutil.copyfile(sample, f"{save_dir}/audio/{name}.wav")
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
                         power=2,
                         norm="slaney",
                         mel_scale="slaney")
    
    np.save(f"{save_dir}/mel/{name}", mel.cpu().numpy())