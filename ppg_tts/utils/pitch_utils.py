import numpy as np
from loguru import logger
from scipy.interpolate import interp1d

def convert_continuos_f0(f0):
    """CONVERT F0 TO CONTINUOUS F0

    Args:
        f0 (ndarray): original f0 sequence with the shape (T)

    Return:
        (ndarray): continuous f0 with the shape (T)
    """
    # get start and end of f0
    if (np.isnan(f0)).all():
        logger.warning("all of the f0 values are NaNs.")
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