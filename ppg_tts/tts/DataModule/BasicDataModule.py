import lightning as L
from typing import Optional
from pathlib import Path
from torch.utils.data.dataloader import DataLoader
from ...dataset import PersoCollateFn, ExtendDataset

class BasicDataModule(L.LightningDataModule):
    def __init__(self, 
                 data_dir: str="./data",
                 batch_size: int=16,
                 no_ctc: bool=False,
                 ppg_sparse: str=None,
                 sparse_coeff: Optional[int | float]=None):
        super().__init__()
        self.data_dir = data_dir
        self.train_dir = Path(data_dir) / "train"
        self.val_dir = Path(data_dir) / "val"
        self.test_dir = Path(data_dir) / "test"
        self.pred_dir = "data/spk_sanity"
        self.batch_size = batch_size
        self.no_ctc = no_ctc
        self.ppg_sparse = ppg_sparse
        self.sparse_coeff = sparse_coeff

        from loguru import logger
        logger.info(f"\nTraining dir: {self.train_dir}\nVal dir: {self.val_dir}\nTest_dir: {self.test_dir}")

    def setup(self, stage: str):
        if stage == 'fit':
            self.train = ExtendDataset(
                data_dir=self.train_dir,
                no_ctc=self.no_ctc,
                ppg_sparse=self.ppg_sparse,
                sparse_coeff=self.sparse_coeff,
            )
            self.val = ExtendDataset(
                data_dir=self.val_dir,
                no_ctc=self.no_ctc,
                ppg_sparse=self.ppg_sparse,
                sparse_coeff=self.sparse_coeff,
            )
        elif stage == 'test':
            self.test = ExtendDataset(
                data_dir=self.test_dir,
                no_ctc=self.no_ctc,
                ppg_sparse=self.ppg_sparse,
                sparse_coeff=self.sparse_coeff,
            )
        elif stage == 'predict':
            self.predict = ExtendDataset(
                data_dir=self.pred_dir,
                no_ctc=self.no_ctc,
                ppg_sparse=self.ppg_sparse,
                sparse_coeff=self.sparse_coeff,
            )

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          num_workers=8,
                          collate_fn=PersoCollateFn,
                          shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size=self.batch_size,
                          num_workers=8,
                          collate_fn=PersoCollateFn)
    
    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size=1,
                          num_workers=4,
                          collate_fn=PersoCollateFn)
    
    def predict_dataloader(self):
        return DataLoader(self.predict,
                          batch_size=1,
                          num_workers=4,
                          collate_fn=PersoCollateFn)

class LibriTTSRDataModule(BasicDataModule):
    def __init__(self, 
                 data_dir: str="./data",
                 batch_size: int=16,
                 no_ctc: bool=False):
        super().__init__(data_dir,
                         batch_size,
                         no_ctc)
        
        self.train_dir = Path(data_dir) / "train-clean-100"
        self.val_dir = Path(data_dir) / "dev-clean"
        self.test_dir = Path(data_dir) / "test-clean"
