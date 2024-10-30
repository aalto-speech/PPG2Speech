import torch
import torchaudio
from loguru import logger
from pathlib import Path
from torchaudio.transforms import Resample, MelSpectrogram
from typing import Tuple, Dict
from torch.utils.data import Dataset
from kaldiio import ReadHelper

class PersoDatasetBasic(Dataset):
    def __init__(self, data_dir: str, target_sr: int=22050):
        self.data_dir = data_dir
        self.wav_scp_path = Path(self.data_dir, "wav.scp")
        self.text_path = Path(self.data_dir, "text")

        logger.info(f"Reading dataset from {data_dir}")

        with open(self.wav_scp_path, "r") as wav_reader:
            self.wav_files = {pair.split(" ")[0]: pair.split(" ")[1].strip("\n") \
                              for pair in wav_reader.readlines()}

        with open(self.text_path, "r") as text_reader:
            self.text_files = {pair.split(" ")[0]: " ".join(pair.split(" ")[1:]).strip("\n") \
                               for pair in text_reader.readlines()}

        self.id2key = {i: key for i, key in enumerate(self.wav_files)}

        self.resampler = Resample(orig_freq=44100, new_freq=target_sr)

    def __getitem__(self, index: int) -> Tuple[str, Dict]:
        if index >= len(self.id2key):
            raise IndexError
        key = self.id2key[index]

        wav = self.wav_files[key]
        text = self.text_files[key]

        waveform, _ = torchaudio.load(wav)

        waveform = self.resampler(waveform)

        return {"key": key, "feature": waveform, "text": text}
    
    def __len__(self) -> int:
        return len(self.id2key)
    

class PersoDatasetWithConditions(PersoDatasetBasic):
    def __init__(self, data_dir):
        super().__init__(data_dir)
        self.ppg_path = Path(data_dir, "ppg.scp")
        self.spk_emb_path = Path(data_dir, "embedding.scp")
        self.log_F0_path = Path(data_dir, "log_f0.scp")

        self.ppgs = self._read_scp_ark(self.ppg_path)
        self.spk_embs = self._read_scp_ark(self.spk_emb_path)
        self.log_F0 = self._read_scp_ark(self.log_F0_path)

        self.melspec = MelSpectrogram(sample_rate=22050,
                                      n_fft=1024,
                                      win_length=1024,
                                      hop_length=256,
                                      f_min=125,
                                      f_max=7600)


    def __getitem__(self, index: int) -> Tuple:
        if index >= len(self.id2key):
            raise IndexError
        key = self.id2key[index]

        wav = self.wav_files[key]
        text = self.text_files[key]

        waveform, _ = torchaudio.load(wav)
        
        waveform = self.resampler(waveform)

        mel = self.melspec(waveform)

        energy = torch.sum(mel ** 2, dim=-1)

        return {"key": key,
                "feature": waveform,
                "text": text,
                "melspectrogram": mel,
                "ppg": torch.Tensor(self.ppgs[key]),
                "spk_emb": torch.Tensor(self.spk_embs[key]),
                "log_F0": torch.Tensor(self.log_F0[key]),
                "energy": energy}

    def __len__(self) -> int:
        return super().__len__()
    
    def _read_scp_ark(self, scp_path: Path):
        key2feat = {}
        with ReadHelper(f"scp:{scp_path}") as reader:
            for key, array in reader:
                key2feat[key] = array
        
        return key2feat

if __name__ == "__main__":
    ds = PersoDatasetWithConditions("./data/debug")
    for d in ds:
        print(d["melspectrogram"].shape)
