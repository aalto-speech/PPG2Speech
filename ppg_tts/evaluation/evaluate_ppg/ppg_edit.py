import fastdtw
import torch
import numpy as np
import random
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
    
    def _dtw_align(self, ppg_hyp: np.ndarray, text_seq: List[str]) -> defaultdict:
        """
        Compute Alignment betwenn PPG hypothesis with text_seq
        Args:
            ppg_hyp: shape (T,)
            text_seq: List[str] of length L
        """
        _, path = fastdtw.fastdtw(ppg_hyp, text_seq, 10, self.dist_func)

        alignment_dict = defaultdict(list)
        start = None
        prev_seq2 = None
        
        for seq1, seq2 in path:
            if prev_seq2 is None or seq2 != prev_seq2:
                if prev_seq2 is not None:
                    alignment_dict[prev_seq2].append((start, seq1))
                start = seq1
            prev_seq2 = seq2
        
        if prev_seq2 is not None:
            alignment_dict[prev_seq2].append((start, seq1 + 1))

        return alignment_dict
    
    def _rebuild_text_with_replace(self, orig_text: str, new_text_seq: List[str], offset: int) -> str:
        reconstruct = []
        curr = 0

        for c in orig_text:
            if c in ' \n':
                reconstruct.append(c)
            else:
                reconstruct.append(self.i2c[new_text_seq[curr + offset]])
                curr += 1
        
        return "".join(reconstruct)
    
    def edit_ppg(self, ppg: np.ndarray, text: str) -> Tuple[np.ndarray, str, Tuple]:
        """
        This function randomly select a frame range and an index.
        Move the dominate probability to the new index
        Args: 
            ppg: torch.Tensor of shape (T, C)
            text: the reference text
        Returns:
            edited ppg
        """
        hyp = ppg.argmax(axis=-1)
        text_seq = [self.c2i[char] for char in text if char not in ' \n']
        OFFSET = 0

        # Add optional silence based on the hypothesis
        if hyp[0] == 1:
            text_seq.insert(0, 1)
            OFFSET = 1
        
        if hyp[-1] == 1:
            text_seq.append(1)

        alignments = self._dtw_align(hyp, text_seq)

        candidates = [key for key in alignments \
                      if key >= OFFSET and 3 <= text_seq[key] <= 31
                     ]

        src_char_idx = random.choice(candidates)
        src_char = text_seq[src_char_idx]
    
        # Randomly select another distinct number from the full range 3-31
        replace = random.choice([x for x in range(3, 32) if x != text_seq[src_char_idx]])

        src_char_start, src_char_end = alignments[src_char_idx][0]

        text_seq[src_char_idx - OFFSET] = replace

        new_ppg = ppg.copy()
        new_ppg.setflags(write=True)

        new_ppg[src_char_start:src_char_end+1, replace] = ppg[src_char_start:src_char_end+1, src_char]
        new_ppg[src_char_start:src_char_end+1, src_char] = 0.0

        return new_ppg, self._rebuild_text_with_replace(text, text_seq, OFFSET), alignments[src_char_idx][0]

if __name__ == '__main__':
    from kaldiio import load_scp
    from ...interpretate.ppg_visualization import visualize_multiple_ppg

    d = load_scp('data/spk_sanity/ppg.scp')

    editor = PPGEditor('data/spk_sanity/phones.txt')

    ppg = d['01_test_0014']

    text = "oluiden myynti laski hieman"

    new_ppg, new_text = editor.edit_ppg(ppg, text)

    visualize_multiple_ppg(
        [ppg[:, :32], new_ppg[:, :32]],
        ['origin', 'edited'],
        save_path='ppg_tts/evaluation/evaluate_ppg/test.png',
        y_labels_map=editor.i2c,
    )

    print(text)
    print(new_text)
    

