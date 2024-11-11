import torch
import torchaudio
from loguru import logger
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from torchaudio.transforms import Resample
from speechbrain.lobes.models.HifiGAN import mel_spectogram
from typing import Tuple, Dict, List
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

    def __getitem__(self, index: int) -> Dict:
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
    def __init__(self, data_dir, no_ctc: bool=False):
        super().__init__(data_dir)
        if no_ctc:
            self.ppg_path = Path(data_dir, "ppg_no_ctc.scp")
        else:
            self.ppg_path = Path(data_dir, "ppg.scp")
        self.spk_emb_path = Path(data_dir, "embedding.scp")
        self.log_F0_path = Path(data_dir, "log_f0.scp")
        self.v_flag = Path(data_dir, "voiced.scp")

        self.ppgs = self._read_scp_ark(self.ppg_path)
        self.spk_embs = self._read_scp_ark(self.spk_emb_path)
        self.log_F0 = self._read_scp_ark(self.log_F0_path)
        self.v_flag = self._read_scp_ark(self.v_flag)


    def __getitem__(self, index: int) -> Dict:
        if index >= len(self.id2key):
            raise IndexError
        key = self.id2key[index]

        wav = self.wav_files[key]
        text = self.text_files[key]

        waveform, _ = torchaudio.load(wav)
        
        waveform = self.resampler(waveform)

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
            audio=waveform,
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
            audio=waveform,
            power=2,
            norm="slaney",
            mel_scale="slaney"
        )

        return {"key": key,
                "feature": waveform,
                "text": text,
                "melspectrogram": mel.squeeze(),
                "ppg": torch.from_numpy(self.ppgs[key].copy()),
                "spk_emb": torch.from_numpy(self.spk_embs[key].copy()),
                "log_F0": torch.from_numpy(self.log_F0[key].copy()),
                "energy": energy.squeeze(),
                "v_flag": torch.from_numpy(self.v_flag[key].copy())}

    def __len__(self) -> int:
        return super().__len__()
    
    def _read_scp_ark(self, scp_path: Path):
        key2feat = {}
        with ReadHelper(f"scp:{scp_path}") as reader:
            for key, array in reader:
                key2feat[key] = array
        
        return key2feat


def PersoCollateFn(batch_lst: List[Dict]) -> Dict[str, torch.Tensor]:
    def _pad_and_batch(key: str):
        items = [d[key].squeeze() for d in batch_lst]

        if key == "melspectrogram" or key == "energy":
            items = [item.transpose(0, 1) for item in items]

        if key == "v_flag":
            items = [item.float() for item in items]

        lengths = torch.tensor([item.size(0) for item in items]).unsqueeze(1)

        batch_tensor = pad_sequence(items, batch_first=True)

        max_len = batch_tensor.size(1)

        max_len_range = torch.arange(max_len).unsqueeze(0)

        mask = lengths <= max_len_range

        return batch_tensor, mask, lengths.squeeze(1)

    mel_batch, mel_mask, _ = _pad_and_batch("melspectrogram")
    ppg_batch, ppg_mask, ppg_length = _pad_and_batch("ppg")
    spk_emb_batch, spk_emb_mask, _ = _pad_and_batch("spk_emb")
    log_F0_batch, log_F0_mask, log_F0_length = _pad_and_batch("log_F0")
    energy_batch, energy_mask, energy_length = _pad_and_batch("energy")
    vflag_batch, _, _ = _pad_and_batch("v_flag")
    keys = [item['key'] for item in batch_lst]

    return {"mel": mel_batch.float(),
            "mel_mask": mel_mask,
            "ppg": ppg_batch.float(),
            "ppg_mask": ppg_mask,
            "ppg_len": ppg_length,
            "spk_emb": spk_emb_batch.float(),
            "spk_emb_mask": spk_emb_mask,
            "log_F0": log_F0_batch.float(),
            "log_F0_mask": log_F0_mask,
            "log_F0_len": log_F0_length,
            "energy": energy_batch.float(),
            "energy_mask": energy_mask,
            "energy_len": energy_length,
            "v_flag": vflag_batch,
            "keys": keys}