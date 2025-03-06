import torch
import torchaudio
from typing import Optional
from loguru import logger
from pathlib import Path
from speechbrain.lobes.models.HifiGAN import mel_spectogram
from torch.utils.data import Dataset
from kaldiio import ReadHelper
from torch.nn.functional import softmax

@torch.no_grad()
def sparse_topK(nn_out: torch.Tensor, k: Optional[int]) -> torch.Tensor:
    """
    Args:
        nn_out: un-softmax output of shape (B, T, C)
        k: threshold of top k
    Returns:
        SPPG
    """
    if k is not None:
        assert type(k) is int, f"k should be an integer in `sparse_topk` function, got {type(k)} instead"
    else:
        k = 3
    B, T, _ = nn_out.shape
    prob = softmax(nn_out, dim=-1)
    topk_prob, topk_idx = prob.topk(k=k, dim=-1)
    sppg = torch.zeros_like(prob)

    b_idx = torch.arange(0, B, dtype=torch.long).view(B, 1, 1)
    t_idx = torch.arange(0, T, dtype=torch.long).view(1, T, 1)
    sppg[b_idx, t_idx, topk_idx] = topk_prob / topk_prob.sum(dim=-1, keepdim=True)

    return sppg

@torch.no_grad()
def sparse_topK_percent(nn_out: torch.Tensor, k: Optional[float]) -> torch.Tensor:
    """
    Args:
        nn_out: un-softmax output of shape (B, T, C)
        k: threshold of top k percentage
    Returns:
        SPPG
    """
    if k is not None:
        assert type(k) is float, f"k should be an float between 0-1 in `sparse_topK_percent` function, got {type(k)} instead"
    else:
        k = 0.95
    B, T, E = nn_out.shape

    prob_tensor = softmax(nn_out, dim=-1)

    # Sort each frame along the E dimension in descending order
    sorted_prob, sorted_indices = torch.sort(prob_tensor, dim=-1, descending=True)

    # Compute the cumulative sum of the sorted probabilities
    cumsum = torch.cumsum(sorted_prob, dim=-1)

    # Find the index where the cumulative sum first meets or exceeds k
    mask = cumsum >= k
    indices = torch.argmax(mask.int(), dim=-1)

    # Create a mask for the probabilities to keep
    # Expand indices to shape (B, T, E) for broadcasting
    indices_expanded = indices.unsqueeze(-1).expand(B, T, E)

    # Create a range tensor for the E dimension
    e_range = torch.arange(E, device=prob_tensor.device).view(1, 1, E).expand(B, T, E)

    # Create the mask where e_range <= indices_expanded
    keep_mask = e_range <= indices_expanded

    # Apply the mask to the sorted probabilities
    sparse_sorted_prob = sorted_prob * keep_mask

    # Revert the sorting to restore the original order
    _, reverse_indices = torch.sort(sorted_indices, dim=-1)
    sparse_prob = torch.gather(sparse_sorted_prob, dim=-1, index=reverse_indices)

    return sparse_prob / torch.sum(sparse_prob, dim=-1, keepdim=True)

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
    
    def _read_scp_ark(self, scp_path: Path):
        key2feat = {}
        with ReadHelper(f"scp:{scp_path}") as reader:
            for key, array in reader:
                key2feat[key] = array
        
        return key2feat

class ExtendDataset(BaseDataset):
    def __init__(self, data_dir: str, target_sr: int=22050, 
                 no_ctc: bool=True, ppg_sparse: str=None, 
                 sparse_coeff: Optional[int | float]=None):
        super().__init__(data_dir, target_sr)
        self.no_ctc = no_ctc

        # flag = '_no_ctc' if self.no_ctc else ''
        flag = '_nn_lsm0.2'
        
        self.ppg_path = Path(data_dir, f"ppg{flag}.scp")
        self.spk_emb_path = Path(data_dir, "embedding.scp")
        self.log_F0_path = Path(data_dir, "log_f0.scp")
        self.v_flag = Path(data_dir, "voiced.scp")

        self.ppgs = self._read_scp_ark(self.ppg_path)
        self.spk_embs = self._read_scp_ark(self.spk_emb_path)
        self.log_F0 = self._read_scp_ark(self.log_F0_path)
        self.v_flag = self._read_scp_ark(self.v_flag)

        self.ppg_sparse = ppg_sparse
        self.sparse_coeff = sparse_coeff

        if ppg_sparse is not None:
            match ppg_sparse:
                case 'topk': self.sparse_func = sparse_topK
                case 'percentage': self.sparse_func = sparse_topK_percent
                case _: raise ValueError('Choose PPG sparse method from topK or percentage')

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

        if self.ppg_sparse is not None:
            ppg = self.sparse_func(ppg.unsqueeze(0), self.sparse_coeff).squeeze(0)

        return {"key": key,
                "feature": wav,
                "text": None,
                "melspectrogram": mel.squeeze(),
                "ppg": ppg,
                "spk_emb": torch.from_numpy(self.spk_embs[key].copy()),
                "log_F0": torch.from_numpy(self.log_F0[key].copy()),
                "energy": energy.squeeze(),
                "v_flag": torch.from_numpy(self.v_flag[key].copy())}
