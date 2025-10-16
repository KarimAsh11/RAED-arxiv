import logging
import torch
import pytorch_lightning as pl
import bitsandbytes as bnb

from functools import wraps
from typing import Any
from transformers import TextGenerationPipeline
from torch.optim import RAdam, AdamW
from transformers import Adafactor
from transformers import get_linear_schedule_with_warmup
from src.models.utils import skip_on_OOM
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import pickle
from transformers import StoppingCriteria, StoppingCriteriaList

class StopOnToken(StoppingCriteria):
    def __init__(self, stop_id):
        self.stop_id = stop_id

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] == self.stop_id

class EmergePLModule(pl.LightningModule):
    def __init__(self, conf, model, tokenizer, *args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(conf)
        self.lr = 1e-6
        self.weight_decay=0.0
        self.no_weight_decay_params = ["bias", "layer_norm.weight"]

        self.model = model
        # self.model.config.n_positions = 1024
        self.tokenizer = tokenizer
        self.pipeline = TextGenerationPipeline(
                model=self.model, tokenizer=self.tokenizer
            )

        self.num_training_steps=conf.train.lr_scheduler.num_training_steps
        self.num_warmup_steps=conf.train.lr_scheduler.num_warmup_steps
        self.gen_params = conf.train.generation_params
        self.reset_validation()
        self.reset_test()
        self.loss = CrossEntropyLoss()
        self.target = conf.data.target
        self.bad_titles = False
        bad_titles_path = None
        self.bad_words={}
        with open(bad_titles_path, 'rb') as f:
            bad_titles = pickle.load(f)
        for q_id, cands in bad_titles.items():
            self.bad_words[q_id] = self.tokenizer(cands, add_special_tokens=False).input_ids

        self.title_terminators = [
                            self.tokenizer.eos_token_id,
                            self.tokenizer.convert_tokens_to_ids("<def>")
                        ]
        print("TITLE TERMINATORS: ", self.title_terminators)
        self.stopping_criteria = StoppingCriteriaList([StopOnToken(self.tokenizer.convert_tokens_to_ids("<def>"))])
        self.stopping_criteria_eos = StoppingCriteriaList([StopOnToken(self.tokenizer.eos_token_id)])

        self.pipeline = TextGenerationPipeline(
                model=self.model, tokenizer=self.tokenizer, device=self.model.device
            )
        self.pipeline.device = self.model.device
        self.emerge_pred = conf.train.emerge_mode

    def forward(self, batch, **kwargs) -> dict:
        """
        Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.

        """
        if 'ids' in batch.keys(): 
            idx = batch.pop('ids')

        # decoded_context_input_ids = self.tokenizer.batch_decode(
        #     batch["context_input_ids"],
        #     skip_special_tokens=False,
        #     clean_up_tokenization_spaces=True,
        # )

        # prompt_len = batch['token_context'].size(1)
        # decoded_labels = self.tokenizer.batch_decode(
        #     batch["target_input_ids"][:, prompt_len:],
        #     skip_special_tokens=False,
        #     clean_up_tokenization_spaces=True,
        # )

        # decoder_input_ids = batch.get("target_input_ids", None)
        # decoder_input_ids[decoder_input_ids == 0] = -100

        out = self.model(
            input_ids=batch["context_input_ids"],
            attention_mask=batch["context_attention_mask"],
            # decoder_attention_mask=batch['target_attention_mask'],
            labels=batch["target_input_ids"],
            **kwargs,
        )

        loss = out.loss
        out_dict = {
            "loss": loss,
            "logits": out.logits,
            "probabilities": out.logits.softmax(dim=-1),
            "predictions": out.logits.argmax(dim=-1),
        }

        return out_dict
            

    @skip_on_OOM(empty_grad=False)
    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        forward_output = self.forward(batch, return_dict=True)
        loss = forward_output["loss"]
        self.log(
            "training_loss",
            loss,
            batch_size=batch["context_input_ids"].size(0),
            prog_bar=True
        )
        cur_lr = self.optimizer.param_groups[0]['lr']
        self.log("lr", cur_lr, prog_bar=True, on_step=True)
        return loss

    def _predict(self, batch: dict):
        """Use the pipeline to generate predictions."""

        # print('batch[input_text]', batch['input_text'])
        preds = [
            out[0]["generated_text"]  # type: ignore
            for out in self.pipeline(
                batch['input_text'],
                return_full_text=False,
                clean_up_tokenization_spaces=True,
                max_new_tokens=self.gen_params.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )  # type: ignore
        ]
        return preds 

    def test_step(self, batch: dict, batch_idx: int) -> None:
        if 'ids' in batch.keys(): 
            idx = batch.pop('ids')
        forward_output = self.forward(batch, return_dict=True)
        test_loss = forward_output["loss"]
        self.log(
            "test_loss",
            test_loss,
            batch_size=batch["context_input_ids"].size(0),
            prog_bar=True
        )
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=batch['context_input_ids'],
                attention_mask=batch['context_attention_mask'],
                num_beams=self.gen_params.num_beams,
                min_length=self.gen_params.min_length,
                max_new_tokens=self.gen_params.max_new_tokens,
            )

            decoded_out = self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            decoded_gold = batch['gold_definition']

            self.test_predictions+=decoded_out
            self._test_gold+=decoded_gold
            self._test_ids+=idx
            
            return test_loss
        
    def validation_tempel_step(self, batch: dict, batch_idx: int) -> None:
        if 'ids' in batch.keys(): 
            idx = batch.pop('ids')
        title_start_ids = batch["title_starts"]
        wikidata_ids = batch["wikidata"][0]
        bad_words = []
        if wikidata_ids in self.bad_words:
            bad_words = self.bad_words[wikidata_ids]

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=batch['context_input_ids'],
                attention_mask=batch['context_attention_mask'],
                num_beams=self.gen_params.num_beams,
                min_length=self.gen_params.min_length,
                max_new_tokens=self.gen_params.max_new_tokens,
            )

            if self.target == 'title_def':
                spec_decoded_out = self.tokenizer.batch_decode(
                    outputs,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
                decoded_out = [o.replace('</s>', "").replace('<unk>', "").replace('<pad>', "") for o in spec_decoded_out]
            else:
                decoded_out = self.tokenizer.batch_decode(
                    outputs,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                
            decoded_gold = batch['gold_definition']

            self.val_predictions+=decoded_out
            self._validation_gold+=decoded_gold
            self._validation_ids+=idx

            val_loss = 0.0
            return val_loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        if 'ids' in batch.keys(): 
            idx = batch.pop('ids')
    
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=batch['token_context'],
                attention_mask=batch['token_context_mask'],
                num_beams=self.gen_params.num_beams,
                min_length=self.gen_params.min_length,
                max_new_tokens=self.gen_params.max_new_tokens,
                eos_token_id=[self.tokenizer.eos_token_id],
                stopping_criteria=self.stopping_criteria_eos,
            )

            if self.target == 'title_def':
                prompt_len = batch['token_context'].size(1)
                outputs = outputs[:, prompt_len:]
                spec_decoded_out = self.tokenizer.batch_decode(
                    outputs,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
                decoded_out = [o.replace('<|endoftext|>', "").replace('<|unktext|>', "").replace('<|padtext|>', "") for o in spec_decoded_out]

            else:
                prompt_len = batch['token_context'].size(1)
                outputs = outputs[:, prompt_len:]
                decoded_out = self.tokenizer.batch_decode(
                    outputs,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )

            decoded_gold = batch['gold_definition']

            self.val_predictions+=decoded_out
            self._validation_gold+=decoded_gold
            self._validation_ids+=idx

            val_loss = 0.0
            return val_loss
        
    def test_step(self, batch: dict, batch_idx: int) -> None:
        if 'ids' in batch.keys(): 
            idx = batch.pop('ids')

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=batch['token_context'],
                attention_mask=batch['token_context_mask'],
                num_beams=self.gen_params.num_beams,
                min_length=self.gen_params.min_length,
                max_new_tokens=self.gen_params.max_new_tokens,
            )

            if self.target == 'title_def':
                prompt_len = batch['token_context'].size(1)
                outputs = outputs[:, prompt_len:]
                spec_decoded_out = self.tokenizer.batch_decode(
                    outputs,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
                decoded_out = [o.replace('<|endoftext|>', "").replace('<|unktext|>', "").replace('<|padtext|>', "") for o in spec_decoded_out]

            else:
                decoded_out = self.tokenizer.batch_decode(
                    outputs,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                
            decoded_gold = batch['gold_definition']

            self.test_predictions+=decoded_out
            self._test_gold+=decoded_gold
            self._test_ids+=idx
            
            test_loss = 0.0
            return test_loss
        

    def configure_optimizers(self):
        """
        FROM PYTORCH LIGHTNING DOCUMENTATION

        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Return:
            Any of these 6 options.

            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        print("lr: ", self.lr)
        print("num_warmup_steps: ", self.num_warmup_steps)
        print("num_training_steps: ", self.num_training_steps)

        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        return self.optimizer

    def reset_validation(self) -> None:
        self.val_predictions = []
        self._validation_gold = []
        self._validation_ids = []

    def reset_test(self) -> None:
        self.test_predictions = []
        self._test_gold = []
        self._test_ids = []