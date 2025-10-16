from pytorch_lightning.callbacks import Callback
import torch
from tqdm import tqdm
from src.models.utils import skip_on_OOM
import csv
import random
import evaluate
import json
import numpy as np
from torch.nn import CrossEntropyLoss

class PredictCallback(Callback):
    def __init__(self, output_val_path, output_test_path):
        self.output_test_path = output_test_path
        self.output_val_path  = output_val_path
        self.max_print = 25

    def on_validation_epoch_end(self, trainer, pl_module):
        print("EVALUATING Validation Set ...")
        pl_module.eval()

        self.mention_extra_contexts = trainer.datamodule.val_dataset.mention_extra_contexts
        with open(self.output_val_path, 'w') as f:
            for k in tqdm(range(len(trainer.datamodule.val_dataset)), desc="Computing Predictions", total=len(trainer.datamodule.val_dataset)):
                sample_id = k
                sample = trainer.datamodule.val_dataset[sample_id].copy()                
                val_preds = pl_module.val_predictions[sample_id]
                val_gold = pl_module._validation_gold[sample_id]
                sample["prediction"] = val_preds
                sample["gold"] = val_gold
                f.write(json.dumps(sample) + '\n')


    def on_test_epoch_end(self, trainer, pl_module):
        print("EVALUATING Test Set...")
        pl_module.eval()
        self.mention_extra_contexts = trainer.datamodule.test_dataset.mention_extra_contexts
        with open(self.output_test_path, 'w') as f:
            for k in tqdm(range(len(trainer.datamodule.test_dataset)), desc="Computing Predictions", total=len(trainer.datamodule.test_dataset)):
                sample_id = k
                sample = trainer.datamodule.test_dataset[sample_id].copy()
                test_preds = pl_module.test_predictions[sample_id]
                test_gold = pl_module._test_gold[sample_id]
                sample["prediction"] = test_preds
                sample["gold"] = test_gold
                f.write(json.dumps(sample) + '\n')
                