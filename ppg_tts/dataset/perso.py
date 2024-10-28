import torchaudio
import pathlib
from torchaudio.transforms import Resample
from typing import Tuple, Dict
from torch.utils.data import Dataset, DataLoader

class PersoDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.wav_scp_path = pathlib.Path(self.data_dir, "wav.scp")
        self.text_path = pathlib.Path(self.data_dir, "text")

        with open(self.wav_scp_path, "r") as wav_reader:
            self.wav_files = {pair.split(" ")[0]: pair.split(" ")[1].strip("\n") \
                              for pair in wav_reader.readlines()}

        with open(self.text_path, "r") as text_reader:
            self.text_files = {pair.split(" ")[0]: " ".join(pair.split(" ")[1:]).strip("\n") \
                               for pair in text_reader.readlines()}

        self.id2key = {i: key for i, key in enumerate(self.wav_files.keys())}

        self.resampler = Resample(orig_freq=44100, new_freq=16000)

    def __getitem__(self, index) -> Tuple[int, Dict]:
        key = self.id2key[index]

        wav = self.wav_files[key]
        text = self.text_files[key]

        waveform, _ = torchaudio.load(wav)
        
        waveform = self.resampler(waveform)

        return key, {"feature": waveform, "text": text}
    
    def __len__(self) -> int:
        return len(self.id2key)
    

class PersoDatasetWithConditions(PersoDataset):
    def __init__(self, data_dir):
        super().__init__(data_dir)
        