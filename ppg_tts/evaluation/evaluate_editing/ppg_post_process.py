import sys
import kaldiio
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from typing import List, Optional
from pathlib import Path

class CD2CIMapper:
    def __init__(self, phone_lst: Optional[str | Path]):
        self.CDPhone2id = dict()
        self.CDPhone2CIPhone = dict()
        self.CIPhone2CDPhoneIdxs = defaultdict(set)
        self.numCIPhone = 0
        self.CIPhone2idx = dict()

        if phone_lst is not None:
            self.read(phone_lst=phone_lst)

    def read(self, phone_lst: str | Path):
        with open(phone_lst, 'r') as reader:
            CDPhones = reader.readlines()

        for phone in CDPhones:
            p, i = phone.strip(' \n').split(' ')
            self.CDPhone2id[p] = int(i)

            cip = p.split('_')[0]
            self.CDPhone2CIPhone[p] = cip
            self.CIPhone2CDPhoneIdxs[cip].add(int(i))

        self.numCIPhone = len(self.CIPhone2CDPhoneIdxs)

        for i, ci_phone in enumerate(self.CIPhone2CDPhoneIdxs.keys()):
            self.CIPhone2idx[ci_phone] = i

    def getCIPhones(self, CDPhone: str) -> str:
        return self.CDPhone2CIPhone[CDPhone]

    def getCDIndex(self, CDPhone: str) -> int:
        return self.CDPhone2id[CDPhone]

    def getCDIndicesFromCIPhone(self, CIPhone: str) -> List[int]:
        return list(self.CIPhone2CDPhoneIdxs[CIPhone])
    
def sumIndices2Index(source: np.ndarray, target: np.ndarray, source_indices: List[int], target_idx: int):
    """
    Args:
        source: shape of (..., d1)
        target: shape of (..., d2)
        source_indices: list of int
        target_idx: int
    """
    target[..., target_idx] = np.sum(source[..., source_indices], axis=-1)

def CDPPG2CIPPG(CDPPG: str | Path, mapper: CD2CIMapper, output: str | Path):
    cdppg_d = kaldiio.load_ark(CDPPG)

    cippg_d = {}

    for key, cd_ppg in tqdm(cdppg_d):
        t, _ = cd_ppg.shape
        ci_ppg = np.zeros((t, mapper.numCIPhone))

        for ci_p, i in mapper.CIPhone2idx.items():
            sumIndices2Index(cd_ppg, ci_ppg, mapper.getCDIndicesFromCIPhone(ci_p), i)
        
        cippg_d[key] = ci_ppg
    
    kaldiio.save_ark(output, cippg_d, scp=output.replace('.ark', '.scp'))

    
if __name__ == "__main__":
    phone_lst = sys.argv[1]
    ark = sys.argv[2]

    write_ark = Path(ark).parent / "ppg.ark"

    mapper = CD2CIMapper(phone_lst)
    
    CDPPG2CIPPG(ark, mapper, write_ark.absolute().as_posix())