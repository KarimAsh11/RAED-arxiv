from typing import Any, Union, List, Optional

from omegaconf import DictConfig

import torch
from pytorch_lightning.utilities import CombinedLoader
from torch.utils.data import DataLoader, SubsetRandomSampler
import pytorch_lightning as pl
from src.data.EmergeDataset import EntDefDataset

from functools import partial

class EmergePLDataModule(pl.LightningDataModule):
    """
    Baseline DataModule - RAED 
    """

    def __init__(self, conf: DictConfig, tokenizer):
        super().__init__()
        self.fid = conf.model.fid
        self.conf = conf.data
        self.datasets = self.conf.datasets
        self.num_workers = self.conf.num_workers
        self.batch_sizes = self.conf.batch_size
        self.batch_size_val = self.conf.batch_size_val

        self.train_extra_contexts = self.conf.train_extra_contexts
        self.test_extra_contexts = self.conf.test_extra_contexts
        self.shuff_prob = self.conf.shuff_prob
        self.target = self.conf.target

        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: Optional[Sequence[Dataset]] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None
        self.tokenizer = tokenizer

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = EntDefDataset(name=self.datasets.train.name, path=self.datasets.train.data_path, tokenizer=self.tokenizer, mention_extra_contexts=self.train_extra_contexts, target=self.target, shuff_prob=self.shuff_prob)
        self.val_dataset   = EntDefDataset(name=self.datasets.val.name, path=self.datasets.val.data_path, tokenizer=self.tokenizer, mention_extra_contexts=self.test_extra_contexts, target=self.target)
        self.test_dataset  = EntDefDataset(name=self.datasets.test.name, path=self.datasets.test.data_path, tokenizer=self.tokenizer, mention_extra_contexts=self.test_extra_contexts, target=self.target)
        
        print("Train dataset size: ", len(self.train_dataset))
        print("Val dataset size: ", len(self.val_dataset))
        print("Test dataset size: ", len(self.test_dataset))

        if self.fid:
            self.collate_fn = self.train_dataset.collate_fid_fn
        else:
            self.collate_fn = self.train_dataset.collate_decoder_fn

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.train_dataset.data,
            shuffle=True,
            batch_size=self.batch_sizes,
            num_workers=self.num_workers.train,
            pin_memory=True,
            collate_fn=partial(
                self.collate_fn, *args, **kwargs
            ),
        )

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.val_dataset.data,
            shuffle=False,
            batch_size=self.batch_size_val,
            num_workers=self.num_workers.val,
            pin_memory=True,
            collate_fn=partial(
                self.collate_fn, *args, **kwargs
            ),
        )
    
    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.test_dataset.data,
            shuffle=False,
            batch_size=self.batch_size_val,
            num_workers=self.num_workers.test,
            pin_memory=True,
            collate_fn=partial(
                self.collate_fn, *args, **kwargs
            ),
        )
