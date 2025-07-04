import numpy as np
import os
from loguru import logger
from librosa import pyin
from typing import Dict, Tuple
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler

standardizer = StandardScaler()

def normalize_voiced_frames(f0: np.ndarray, flags: np.ndarray) -> np.ndarray:
    voiced_frames = f0[flags == True][:, np.newaxis]

    if voiced_frames.shape[0] == 0:
        return f0

    normalized_frames = standardizer.fit_transform(voiced_frames)

    f0[flags == True] = normalized_frames.squeeze(-1)

    return f0

def extract_f0_from_utterance(utterance: Dict, normalize: bool=False) \
    -> Tuple[str, np.ndarray, np.ndarray]:
    wav = utterance["feature"]
    if not isinstance(wav, np.ndarray):
        wav = wav.numpy()
    foundamental_freq, voiced_flag, _ = pyin(y=wav,
                                             fmin=1e-6,
                                             fmax=8000,
                                             sr=22050,
                                             hop_length=256,
                                             frame_length=1024)

    if normalize:
        foundamental_freq = normalize_voiced_frames(foundamental_freq, voiced_flag)
                
    foundamental_freq = convert_continuos_f0(utterance["key"], foundamental_freq.squeeze())
    foundamental_freq = np.log(foundamental_freq,
                               where=foundamental_freq>0)
    
    logger.info(f"Process {os.getpid()} - {utterance['key']}: log_F0 shape {foundamental_freq.shape}")
    
    return utterance["key"], foundamental_freq, voiced_flag.squeeze().astype(np.float32)

def convert_continuos_f0(key, f0):
    """CONVERT F0 TO CONTINUOUS F0

    Args:
        f0 (ndarray): original f0 sequence with the shape (T)

    Return:
        (ndarray): continuous f0 with the shape (T)
    """
    # get start and end of f0
    if (np.isnan(f0)).all():
        logger.warning(f"{key}: all of the f0 values are NaNs.")
        return np.nan_to_num(f0, nan=1e-6)
    start_f0 = f0[~np.isnan(f0)][0]
    end_f0 = f0[~np.isnan(f0)][-1]

    # padding start and end of f0 sequence
    start_idx = np.where(f0 == start_f0)[0][0]
    end_idx = np.where(f0 == end_f0)[0][-1]
    f0[:start_idx] = start_f0
    f0[end_idx:] = end_f0

    # get non-zero frame index
    nz_frames = np.where(~np.isnan(f0))[0]

    # perform linear interpolation
    f = interp1d(nz_frames, f0[nz_frames])
    cont_f0 = f(np.arange(0, f0.shape[0]))

    return cont_f0
