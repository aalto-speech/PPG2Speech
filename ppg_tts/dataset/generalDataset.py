import torch
import torchaudio
from loguru import logger
from pathlib import Path
from speechbrain.lobes.models.HifiGAN import mel_spectogram
from torch.utils.data import Dataset
from kaldiio import ReadHelper
from torch.nn.functional import interpolate

class BaseDataset(Dataset):
    def __init__(self, data_dir: str, target_sr: int=22050):
        super().__init__()
        self.target_sr = target_sr
        logger.info(f"Reading dataset from {data_dir}")
        with open(f"{data_dir}/wav.scp", "r") as reader:
            wavlist = reader.readlines()
        
        self.key2wav = {}
        self.idx2key = {}
        self.key2idx = {}
        for i, item in enumerate(wavlist):
            key, path = item.split(" ")
            path = path.strip(" \n")
            self.key2wav[key] = path
            self.idx2key[i] = key
            self.key2idx[key] = i

        with open(f"{data_dir}/text", "r") as reader:
            textlist = reader.readlines()

        self.key2text = {}

        for text in textlist:
            key, *words = text.strip('\n').split()
            self.key2text[key] = " ".join(words)

    def __len__(self):
        return len(self.idx2key)
    
    def __getitem__(self, index):
        if index >= len(self.idx2key):
            raise IndexError
        
        key = self.idx2key[index]

        path = self.key2wav[key]

        wav, sr = torchaudio.load(path)

        wav = torchaudio.functional.resample(wav,
                                             orig_freq=sr,
                                             new_freq=self.target_sr)
        
        return key, wav, sr, self.key2text[key]


class ExtendDataset(BaseDataset):
    def __init__(self, data_dir: str, target_sr: int=22050, no_ctc: bool=True):
        super().__init__(data_dir, target_sr)
        self.no_ctc = no_ctc

        flag = '_no_ctc' if self.no_ctc else ''
        
        self.ppg_path = Path(data_dir, f"ppg{flag}.scp")
        self.spk_emb_path = Path(data_dir, "embedding.scp")
        self.log_F0_path = Path(data_dir, "log_f0.scp")
        self.v_flag = Path(data_dir, "voiced.scp")

        self.ppgs = self._read_scp_ark(self.ppg_path)
        self.spk_embs = self._read_scp_ark(self.spk_emb_path)
        self.log_F0 = self._read_scp_ark(self.log_F0_path)
        self.v_flag = self._read_scp_ark(self.v_flag)

    def __getitem__(self, index):
        if index >= len(self.idx2key):
            raise IndexError
        
        key = self.idx2key[index]

        path = self.key2wav[key]

        wav, sr = torchaudio.load(path)

        wav = torchaudio.functional.resample(wav,
                                             orig_freq=sr,
                                             new_freq=self.target_sr)
        
        mel = mel_spectogram(
            sample_rate=22050,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            f_min=0,
            f_max=8000,
            n_mels=80,
            normalized=False,
            compression=True,
            audio=wav,
            power=1,
            norm="slaney",
            mel_scale="slaney"
        )
        
        energy = mel_spectogram(
            sample_rate=22050,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            f_min=0,
            f_max=8000,
            n_mels=80,
            normalized=False,
            compression=True,
            audio=wav,
            power=2,
            norm="slaney",
            mel_scale="slaney"
        )

        ppg = torch.from_numpy(self.ppgs[key].copy())
        T = mel.size(-1)

        ppg = self._interpolate(ppg, T)

        return {"key": key,
                "feature": wav,
                "text": None,
                "melspectrogram": mel.squeeze(),
                "ppg": ppg,
                "spk_emb": torch.from_numpy(self.spk_embs[key].copy()),
                "log_F0": torch.from_numpy(self.log_F0[key].copy()),
                "energy": energy.squeeze(),
                "v_flag": torch.from_numpy(self.v_flag[key].copy())}

    def _read_scp_ark(self, scp_path: Path):
        key2feat = {}
        with ReadHelper(f"scp:{scp_path}") as reader:
            for key, array in reader:
                key2feat[key] = array
        
        return key2feat
    
    def _interpolate(self,
                     x: torch.Tensor,
                     target_length: int) -> torch.Tensor:
        x = x.unsqueeze(0).permute(0, 2, 1)

        x_interpolated = interpolate(x, 
                                     size=target_length, 
                                     mode='linear', 
                                     align_corners=True)

        x_interpolated = x_interpolated.permute(0, 2, 1).squeeze(0)

        return x_interpolated
        