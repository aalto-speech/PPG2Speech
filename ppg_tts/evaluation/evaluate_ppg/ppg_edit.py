import fastdtw
import torch
import numpy as np
import random
from collections import defaultdict
from typing import Tuple, List, Dict

SUBSAMPLING_FACTOR = 3

common_errors = {
    'ä': ('a', 'e'),
    'ö': ('o',),
    'y': ('u', 'e'),
    'r': ('l', 'w'),
    'a': ('ä',),
    'o': ('ö',),
}

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

        self.common_error_ids = {}

        for key in common_errors:
            i_k = self.c2i[key]

            self.common_error_ids[i_k] = tuple([self.c2i[item] for item in common_errors[key]])

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
    
    def _rebuild_text_with_replace(self, orig_text: str, new_text_seq: List[str], offset: int) -> Tuple[str, int]:
        reconstruct = []
        curr = 0

        for c in orig_text:
            if c in ' \n':
                reconstruct.append(c)
            else:
                reconstruct.append(self.i2c[new_text_seq[curr + offset]])
                curr += 1

        new_string = "".join(reconstruct)

        diff_idx = -1

        for i, (c1, c2) in enumerate(zip(orig_text, new_string)):
            if c1 != c2:
                diff_idx = i
                break
        
        if diff_idx < len(orig_text) - 1 and orig_text[diff_idx + 1] != new_string[diff_idx + 1]:
            diff_idx = (diff_idx, diff_idx + 1)
        
        return new_string, diff_idx
    
    def edit_ppg(self, ppg: np.ndarray, text: str) -> Tuple[np.ndarray, Tuple[str, int], Tuple]:
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

        
            # Select character in the error set, replace with specific error
        candidate_indices = list(filter(lambda x: text_seq[x] in self.common_error_ids, range(len(text_seq))))

        index = random.choice(candidate_indices)

        src_char = text_seq[index]
        target_char = random.choice(self.common_error_ids[src_char])

        if index > 0 and src_char == text_seq[index - 1]:
            src_char_start = alignments[index - 1][0][0]
            src_char_end = alignments[index][0][1]
            text_seq[index - 1] = target_char
            text_seq[index] = target_char
        elif index < len(text_seq) - 1 and src_char == text_seq[index + 1]:
            src_char_start = alignments[index][0][0]
            src_char_end = alignments[index + 1][0][1]
            text_seq[index + 1] = target_char
            text_seq[index] = target_char
        else:
            src_char_start, src_char_end = alignments[index][0]
            text_seq[index] = target_char

        new_ppg = ppg.copy()
        new_ppg.setflags(write=True)

        new_ppg[src_char_start:src_char_end+1, target_char] = ppg[src_char_start:src_char_end+1, src_char]
        new_ppg[src_char_start:src_char_end+1, src_char] = 0.0

        return new_ppg, self._rebuild_text_with_replace(text, text_seq, OFFSET), (src_char_start, src_char_end)

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
    

