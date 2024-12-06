import lightning as L
from loguru import logger
from pathlib import Path
from torch.utils.data.dataloader import DataLoader
from ...dataset import PersoDatasetWithConditions, PersoCollateFn

class PersoDataModule(L.LightningDataModule):
    def __init__(self, 
                 data_dir: str="./data",
                 batch_size: int=16,
                 no_ctc: bool=False):
        super().__init__()
        logger.warning(f"{self.__class__.__name__} is deprecated. Please use BasicDataModule for Perso instead.")
        self.data_dir = data_dir
        self.train_dir = Path(data_dir) / "train"
        self.val_dir = Path(data_dir) / "val"
        self.test_dir = Path(data_dir) / "test"
        self.batch_size = batch_size
        self.no_ctc = no_ctc

    def setup(self, stage: str):
        if stage == 'fit':
            self.train = PersoDatasetWithConditions(self.train_dir, self.no_ctc)
            self.val = PersoDatasetWithConditions(self.val_dir, self.no_ctc)
        elif stage == 'test' or stage == 'predict':
            # self.test = PersoDatasetWithConditions(self.val_dir, self.no_ctc)
            self.test = PersoDatasetWithConditions(self.test_dir, self.no_ctc)

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          num_workers=8,
                          collate_fn=PersoCollateFn)
    
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
