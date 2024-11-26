import lightning as L
from loguru import logger
from typing import Dict
from pathlib import Path
from torch.utils.data.dataloader import DataLoader
from ...dataset import PersoCollateFn, VCTKExtend, PersoDatasetWithConditions

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

        if 'perso' in data_dirs:
            self.perso_train_dir = Path(data_dirs['perso']) / 'train'
            self.perso_val_dir = Path(data_dirs['perso']) / 'val'
            self.perso_test_dir = Path(data_dirs['perso']) / 'test'

        if 'vctk' in data_dirs:
            self.vctk_train_dir = Path(data_dirs['vctk']) / 'train'
            self.vctk_val_dir = Path(data_dirs['vctk']) / 'val'
            self.vctk_test_dir = Path(data_dirs['vctk']) / 'test'

    def setup(self, stage):
        if stage == 'fit':
            self.perso_train = PersoDatasetWithConditions(self.perso_train_dir, self.no_ctc)
            self.perso_val = PersoDatasetWithConditions(self.perso_val_dir, self.no_ctc)

            self.vctk_train = VCTKExtend(data_dir=self.vctk_train_dir, no_ctc=self.no_ctc)
            self.vctk_val = VCTKExtend(data_dir=self.vctk_val_dir, no_ctc=self.no_ctc)
        
        elif stage == 'test' or stage == 'predict':
            self.vctk_test = VCTKExtend(data_dir=self.vctk_test_dir, no_ctc=self.no_ctc)
            self.perso_test = PersoDatasetWithConditions(self.perso_test_dir, self.no_ctc)
    
    def train_dataloader(self):
        return [DataLoader(self.perso_train,
                          batch_size=self.batch_size,
                          num_workers=8,
                          collate_fn=PersoCollateFn),
                DataLoader(self.vctk_train,
                          batch_size=self.batch_size,
                          num_workers=8,
                          collate_fn=PersoCollateFn)]
    
    def val_dataloader(self):
        return [DataLoader(self.perso_val,
                          batch_size=self.batch_size,
                          num_workers=8,
                          collate_fn=PersoCollateFn),
                DataLoader(self.vctk_val,
                          batch_size=self.batch_size,
                          num_workers=8,
                          collate_fn=PersoCollateFn)]
    
    def test_dataloader(self):
        return [DataLoader(self.perso_test,
                          batch_size=1,
                          num_workers=4,
                          collate_fn=PersoCollateFn),
                DataLoader(self.vctk_test,
                          batch_size=1,
                          num_workers=4,
                          collate_fn=PersoCollateFn)]
    
    def predict_dataloader(self):
        return [DataLoader(self.perso_test,
                          batch_size=1,
                          num_workers=4,
                          collate_fn=PersoCollateFn),
                DataLoader(self.vctk_test,
                          batch_size=1,
                          num_workers=4,
                          collate_fn=PersoCollateFn)]
        