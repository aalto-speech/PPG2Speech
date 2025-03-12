import fastdtw
import torch
import numpy as np
import re
from collections import defaultdict
from typing import Tuple, List, Dict

SUBSAMPLING_FACTOR = 3

class PPGEditor:
    def __init__(self, phns: str):
        self.c2i = {}
        self.i2c = {}

        with open(phns, 'r') as reader:
            lines = reader.readlines()

        for line in lines:
            c, i = line.strip(' \n').split()
            self.c2i[c] = int(i)
            self.i2c[int(i)] = c

        self.dist_func = lambda x, y: 0 if x == y else 1

    def char_to_idx(self, c: str) -> int:
        return self.c2i.get(c)

    def idx_to_char(self, i: int) -> str:
        return self.i2c.get(i)

    def IdxSeq2Range(self, seq: List[int]) -> Dict[int, List[Tuple]]:
        ranges = defaultdict(list)

        start = 0
        n = len(seq)

        for i in range(1, n + 1):
            if i == n or seq[i] != seq[i - 1]:  # End of a segment
                idx = seq[start]
                ranges[idx].append(
                    (start * SUBSAMPLING_FACTOR, i * SUBSAMPLING_FACTOR)
                )
                start = i  # Update the start of the next segment

        return ranges
    
    def _dtw_align(self, ppg_hyp: np.ndarray, text_seq: List[str]):
        """
        Compute Alignment betwenn PPG hypothesis with text_seq
        Args:
            ppg_hyp: shape (T,)
            text_seq: List[str] of length L
        """
        return fastdtw.fastdtw(ppg_hyp, text_seq, self.dist_func)

    def _remove_punctuation(self, text: str) -> str:
        """Removes punctuation from the given Finnish text."""
        return re.sub(r"[^\w\säöÄÖ]", "", text).lower()
    
    def edit_ppg(self, ppg: np.ndarray, ali_seq: List[int], text: str) -> torch.Tensor:
        """
        This function randomly select a frame range and an index.
        Move the dominate probability to the new index
        Args: 
            ppg: torch.Tensor of shape (T, C)
            ali_seq: List of int, alignment from Kaldi model
            text: the reference text
        Returns:
            edited ppg
        """
        hyp = ppg.argmax(axis=-1)
        text_seq = self._remove_punctuation(text)

        _, path = self._dtw_align(hyp, text_seq)

        print(path)

if __name__ == '__main__':
    sequence = [1, 1, 1, 1, 1, 1, 1, 1, 1, 17, 17, 17, 17, 17, 17, 10, 10, 10, 10, 
                12, 12, 12, 7, 7, 7, 7, 16, 21, 11, 11, 11, 14, 14, 14, 29, 29, 
                13, 13, 29, 29, 29, 29, 16, 16, 17, 14, 14, 14, 14, 23, 23, 22, 
                22, 22, 3, 3, 3, 21, 21, 11, 11, 11, 11, 3, 3, 3, 3, 3, 3, 1, 1, 
                1, 1, 1, 1, 1]
    
    processer = PPGEditor('data/spk_sanity/phones.txt')
    ranges_dict = processer.IdxSeq2Range(sequence)
    print(ranges_dict)    
