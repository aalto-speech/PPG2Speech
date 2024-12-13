import lightning as L
from pathlib import Path
from torch.utils.data.dataloader import DataLoader
from ...dataset import PersoCollateFn, ExtendDataset

class BasicDataModule(L.LightningDataModule):
    def __init__(self, 
                 data_dir: str="./data",
                 batch_size: int=16,
                 no_ctc: bool=False):
        super().__init__()
        self.data_dir = data_dir
        self.train_dir = Path(data_dir) / "train"
        self.val_dir = Path(data_dir) / "val"
        self.test_dir = Path(data_dir) / "test"
        self.batch_size = batch_size
        self.no_ctc = no_ctc

        from loguru import logger
        logger.info(f"\nTraining dir: {self.train_dir}\nVal dir: {self.val_dir}\nTest_dir: {self.test_dir}")

    def setup(self, stage: str):
        if stage == 'fit':
            self.train = ExtendDataset(data_dir=self.train_dir, no_ctc=self.no_ctc)
            self.val = ExtendDataset(data_dir=self.val_dir, no_ctc=self.no_ctc)
        elif stage == 'test' or stage == 'predict':
            self.test = ExtendDataset(data_dir=self.test_dir, no_ctc=self.no_ctc)

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
        return DataLoader(self.test,
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
