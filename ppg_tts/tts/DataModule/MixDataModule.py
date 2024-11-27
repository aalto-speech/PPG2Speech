import lightning as L
from loguru import logger
from typing import Dict
from pathlib import Path
from torch.utils.data.dataloader import DataLoader
from ...dataset import PersoCollateFn, VCTKLibriTTSRExtend, PersoDatasetWithConditions

class MixDataModule(L.LightningDataModule):
    def __init__(self, 
                 data_dirs: Dict[str, str],
                 batch_size: int=16,
                 no_ctc: bool=False):
        super().__init__()
        self.data_dirs = data_dirs
        self.no_ctc = no_ctc
        self.batch_size = batch_size

        logger.info("Only support perso + vctk mix dataset.")

        self.perso = False
        self.vctk = False
        self.librittsr = False

        if 'perso' in data_dirs:
            self.perso = True
            self.perso_train_dir = Path(data_dirs['perso']) / 'train'
            self.perso_val_dir = Path(data_dirs['perso']) / 'val'
            self.perso_test_dir = Path(data_dirs['perso']) / 'test'

        if 'vctk' in data_dirs:
            self.vctk = True
            self.vctk_train_dir = Path(data_dirs['vctk']) / 'train'
            self.vctk_val_dir = Path(data_dirs['vctk']) / 'val'
            self.vctk_test_dir = Path(data_dirs['vctk']) / 'test'

        if 'librittsr' in data_dirs:
            self.librittsr = True
            self.librittsr_train_dir = Path(data_dirs['librittsr']) / 'train'
            self.librittsr_val_dir = Path(data_dirs['librittsr']) / 'val'
            self.librittsr_test_dir = Path(data_dirs['librittsr']) / 'test'

    def setup(self, stage):
        if stage == 'fit':
            if self.perso:
                self.perso_train = PersoDatasetWithConditions(self.perso_train_dir, self.no_ctc)
                self.perso_val = PersoDatasetWithConditions(self.perso_val_dir, self.no_ctc)

            if self.vctk:
                self.vctk_train = VCTKLibriTTSRExtend(data_dir=self.vctk_train_dir, no_ctc=self.no_ctc)
                self.vctk_val = VCTKLibriTTSRExtend(data_dir=self.vctk_val_dir, no_ctc=self.no_ctc)

            if self.librittsr:
                self.librittsr_train = VCTKLibriTTSRExtend(data_dir=self.librittsr_train_dir, no_ctc=self.no_ctc)
                self.librittsr_val = VCTKLibriTTSRExtend(data_dir=self.librittsr_val_dir, no_ctc=self.no_ctc)
        
        elif stage == 'test' or stage == 'predict':
            if self.vctk:
                self.vctk_test = VCTKLibriTTSRExtend(data_dir=self.vctk_test_dir, no_ctc=self.no_ctc)
            if self.perso:
                self.perso_test = PersoDatasetWithConditions(self.perso_test_dir, self.no_ctc)
            if self.librittsr:
                self.librittsr_test = VCTKLibriTTSRExtend(data_dir=self.librittsr_test_dir, no_ctc=self.no_ctc)
    
    def train_dataloader(self):
        dataloader_lst = []
        if self.perso:
            dataloader_lst.append(
                DataLoader(self.perso_train,
                           batch_size=self.batch_size,
                           num_workers=8,
                           collate_fn=PersoCollateFn)
            )
        
        if self.vctk:
            dataloader_lst.append(
                DataLoader(self.vctk_train,
                           batch_size=self.batch_size,
                           num_workers=8,
                           collate_fn=PersoCollateFn)
            )

        if self.librittsr:
            dataloader_lst.append(
                DataLoader(self.librittsr_train,
                           batch_size=self.batch_size,
                           num_workers=8,
                           collate_fn=PersoCollateFn)
            )
        return dataloader_lst
    
    def val_dataloader(self):
        dataloader_lst = []
        if self.perso:
            dataloader_lst.append(
                DataLoader(self.perso_val,
                           batch_size=self.batch_size,
                           num_workers=8,
                           collate_fn=PersoCollateFn)
            )
        
        if self.vctk:
            dataloader_lst.append(
                DataLoader(self.vctk_val,
                           batch_size=self.batch_size,
                           num_workers=8,
                           collate_fn=PersoCollateFn)
            )

        if self.librittsr:
            dataloader_lst.append(
                DataLoader(self.librittsr_val,
                           batch_size=self.batch_size,
                           num_workers=8,
                           collate_fn=PersoCollateFn)
            )
        return dataloader_lst
    
    def test_dataloader(self):
        dataloader_lst = []
        if self.perso:
            dataloader_lst.append(
                DataLoader(self.perso_test,
                           batch_size=1,
                           num_workers=4,
                           collate_fn=PersoCollateFn)
            )
        
        if self.vctk:
            dataloader_lst.append(
                DataLoader(self.vctk_test,
                           batch_size=1,
                           num_workers=4,
                           collate_fn=PersoCollateFn)
            )

        if self.librittsr:
            dataloader_lst.append(
                DataLoader(self.librittsr_test,
                           batch_size=1,
                           num_workers=4,
                           collate_fn=PersoCollateFn)
            )
        return dataloader_lst
    
    def predict_dataloader(self):
        return self.test_dataloader()
        