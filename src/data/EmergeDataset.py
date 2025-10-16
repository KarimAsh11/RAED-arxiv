import os
import json
import pickle
import random
from tqdm import tqdm
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, Union, Optional

import torch
from torch.utils.data import Dataset, IterableDataset

from src.common.log import get_logger
from src.data.data_utils.utils import normalize

from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence
from transformers import BatchEncoding
from transformers import AutoTokenizer, PreTrainedTokenizer


logger = get_logger(__name__, level="INFO")


class EntDefDataset(Dataset):
    def __init__(
        self,
        name: str,
        path: Union[str, os.PathLike, List[str], List[os.PathLike]] = None,
        data: Any = None,
        tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
        mention_map_path: str = None,
        mention_extra_contexts: int = 0,
        def_source: str = 'Wikipedia',
        target: str = 'title',
        device: str = "cuda:0",
        shuffle_prob: float = 0.0,
        **kwargs,
    ):

        if path is None and data is None:
            raise ValueError("Either `path` or `data` must be provided")
        if tokenizer is None: raise ValueError("A tokenizer must be provided")

        # super().__init__()
        self.name = name
        self.project_folder = Path(__file__).parent.parent.parent
        self.path = path

        self.tokenizer = tokenizer
        if isinstance(self.tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)

        self.def_source = def_source
        print("def_source: ", self.def_source)

        self.data = self.load(data_path=self.path, split=self.name)
        self.device = device
        self.max_token_len = 512

        if mention_map_path:
            self.mention_map = pickle.load(open(self.project_folder / mention_map_path, "rb"))
        else:
            self.mention_map = {}

        self.mention_extra_contexts = mention_extra_contexts
        self.target = target
        self.use_masked_entities = False
        self.use_drop = False
        self.use_random_docs = False
        self.use_drop_query = False
        self.use_shuffle = False

        self.shuff_prob = shuffle_prob

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self, index
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        return self.data[index]

    def __repr__(self) -> str:
        return f"Dataset({self.name=}, {self.path=})"

    def load(
        self,
        data_path,
        split,
        *args,
        **kwargs,
    ) -> Any:
        data_path = self.project_folder / data_path
        data=[]
        with open(data_path, "r") as file:
            for i, sample in enumerate(file):
                sample=json.loads(sample)
                if 'candidates_WIKIPEDIA' not in sample: sample['candidates_WIKIPEDIA'] = []

                if self.def_source == 'Wikidata':
                    sample['gold_definition']=sample['gold_definition_wikidata']
                    if 'candidates_WIKIDATA' in sample:
                        sample['candidates']=sample['candidates_WIKIDATA']
                    else:
                        sample['candidates']=sample['candidates_WIKIPEDIA']
                else:
                    sample['gold_definition']=sample['gold_definition_wikipedia']
                    sample['candidates']=sample['candidates_WIKIPEDIA']
                sample['gold_title'] = sample['wikipedia'].replace("_", " ")
                sample['gold_title_def'] = sample['gold_title'] + " <def> " + sample['gold_definition']
                sample['context']=normalize(sample['context'])
                if 'gold_definition_wikidata' in sample:
                    del sample['gold_definition_wikidata']
                if 'gold_definition_wikipedia' in sample:
                    del sample['gold_definition_wikipedia']
                if 'candidates_WIKIDATA' in sample:
                    del sample['candidates_WIKIDATA']
                if 'candidates_WIKIPEDIA' in sample:
                    del sample['candidates_WIKIPEDIA']
                data.append(sample)
        return data

    def mask_entities(self, retrieved_contexts):
        retr_masked = []
        for retr in retrieved_contexts:
            text = retr['text']
            ner = retr['ner']
            ner = sorted(ner, key=lambda x: x['start'], reverse=True)
            masked_text = list(text)
            for entity in ner:
                random_number = random.random()
                if random_number > 0.2:
                    retr_masked.append(retr)
                    continue
                start = entity['start']
                end = entity['end']
                masked_text[start:end] = '[ENT_MASK]'
            retr["text"] = ''.join(masked_text)  
            retr_masked.append(retr)
        return retr_masked

    def prompt_prep(self, context):
        full_prompt =f"{context}"
        return full_prompt
        
    def collate_fn(self, batch: Any, *args, **kwargs) -> Any:
        idx = [sample["id"] for sample in batch]
        context = []
        if self.target == 'title':
            gold_definition=[sample['gold_title'] for sample in batch]
        elif self.target == 'title_def':
            gold_definition=[sample['gold_title_def'] for sample in batch]
        elif self.target == 'definition':
            gold_definition=[sample['gold_definition'] for sample in batch]

        for sample in batch:
            sample_input = "[QRY] " + sample["context"] + " [/QRY]"
            if "candidates_RETRIEVER" in sample.keys():
                retrieved_contexts = sample['candidates_RETRIEVER'][:self.mention_extra_contexts]
            else:
                retrieved_contexts = []

            if self.name == 'train':
                if len(retrieved_contexts) > 0:
                    if self.use_shuffle:
                        random_number = random.random()
                        if random_number <= self.shuff_prob:
                            random.shuffle(retrieved_contexts)

                    if self.use_drop:
                        random_number = random.random()
                        if random_number <= 0.1:
                            random_number = random.randint(0, self.mention_extra_contexts)
                            retrieved_contexts = retrieved_contexts[:random_number] 

                    if self.use_masked_entities:
                        random_number = random.random()
                        if random_number <= 0.2:
                            retrieved_contexts = self.mask_entities(retrieved_contexts)

                    if self.use_drop_query:
                        random_number = random.random()
                        if random_number <= 0.2:
                            sample_input = ""

                    if self.use_random_docs:
                        random_number = random.random()
                        if random_number <= 0.2:
                            random_index = random.randint(0, len(self.data)-1)
                            retrieved_contexts = self.data[random_index]['candidates_RETRIEVER'][:self.mention_extra_contexts]

                    sample_context = ""
                    for context_string in retrieved_contexts:
                        sample_context += f" [CTX] {context_string['text']} [/CTX]."
                else:
                    sample_context = ""

            if self.name != 'train':
                sample_context = ""
                if len(retrieved_contexts) > 0:
                    for context_string in retrieved_contexts:
                        sample_context += f" [CTX] {context_string['text']} [/CTX]."
                else:
                    sample_context = ""

            sample_query = sample_input
            sample_context = sample_context + sample_query
            sample_context = self.prompt_prep(sample_context)
            context.append(sample_context)

        tokenized_context    = self.tokenizer(context, padding=True, truncation=True, max_length=1024, return_tensors="pt")
        tokenized_definition = self.tokenizer(gold_definition, padding=True, truncation=True, max_length=128, return_tensors="pt")
        batch_inputs = {
            "ids": idx,
            "context_input_ids": tokenized_context.input_ids,
            "context_attention_mask": tokenized_context.attention_mask,
            "target_input_ids": tokenized_definition.input_ids,
            "target_attention_mask": tokenized_definition.attention_mask,
            "gold_definition": gold_definition,
            }
        return batch_inputs   
    
    def encode_passages(self, batch_text_passages, max_length):
        passage_ids, passage_masks = [], []
        for k, text_passages in enumerate(batch_text_passages):
            p = self.tokenizer.batch_encode_plus(
                text_passages,
                max_length=max_length,
                padding='longest',
                return_tensors='pt',
                truncation=True
            )
            passage_ids.append(p['input_ids'][None])
            passage_masks.append(p['attention_mask'][None])
        passage_ids = torch.cat(passage_ids, dim=0)
        passage_masks = torch.cat(passage_masks, dim=0)
        passages_input = {
            "input_ids": passage_ids,
            "attention_mask": passage_masks
        }

        return passages_input

    def collate_fid_fn(self, batch: Any, *args, **kwargs) -> Any:
        idx = [sample["id"] for sample in batch]
        wikidata = [sample["wikidata"] for sample in batch]
        title_starts = [sample["gold_title"] + " <def> " for sample in batch]
        context = []
        if self.target == 'title':
            gold_definition=[sample['gold_title'] for sample in batch]
        elif self.target == 'title_def':
            gold_definition=[sample['gold_title_def'] for sample in batch]
        elif self.target == 'definition':
            gold_definition=[sample['gold_definition'] for sample in batch]

        def append_question(example):
            if example['candidates_RETRIEVER'] is None:
                return [example['context']]
            retrieved_contexts = example['candidates_RETRIEVER'][:self.mention_extra_contexts]
            if self.name == 'train':
                if self.use_shuffle:
                    random_number = random.random()
                    if random_number <= self.shuff_prob:
                        random.shuffle(retrieved_contexts)


            return ["[QRY] " + example['context'] + " [/QRY] " + t['text'] for t in retrieved_contexts]
        context = [append_question(example) for example in batch]
        tokenized_definition = self.tokenizer(gold_definition, padding=True, truncation=True, max_length=52, return_tensors="pt")
        tokenized_context = self.encode_passages(context, max_length=512)

        batch_inputs = {
            "ids": idx,
            "context_input_ids": tokenized_context["input_ids"],
            "context_attention_mask": tokenized_context["attention_mask"],
            "target_input_ids": tokenized_definition.input_ids,
            "target_attention_mask": tokenized_definition.attention_mask,
            "gold_definition": gold_definition,
            "title_starts": title_starts,
            "wikidata": wikidata
            }
        
        return batch_inputs   
    
    def dec_tokenize(self, prompt):
        result = self.tokenizer(
                prompt,                 
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,)

        return result
    
    def generate_and_tokenize_prompt(self, context, title_def):
        if self.target == 'title':
            full_prompt =f"Generate a title for the mention of an entity between [DEF] and [/DEF]. ### Context: {context}### Target Entity Title:"
        elif self.target == 'title_def':
            full_prompt =f"Generate a title and definition seperated by <def> for the mention of an entity between [DEF] and [/DEF]. ### Context: {context}### Target Entity Definition:"
        
        full_prompt = self.tokenizer.bos_token + full_prompt

        tokenized_context = self.dec_tokenize(full_prompt)
        tokenized_definition = self.dec_tokenize(title_def)

        tokenized_cd = torch.cat(
            [   
                tokenized_context.input_ids.squeeze(dim=0),
                tokenized_definition.input_ids.squeeze(dim=0),
                torch.tensor(
                    [self.tokenizer.eos_token_id],
                    dtype=tokenized_definition.input_ids.dtype,
                ),
            ],
            dim=0,
        )

        context_mask = pad(
            torch.ones_like(tokenized_context.input_ids.squeeze(dim=0)),
            (
                0,
                tokenized_cd.size(0)
                - tokenized_context.input_ids.squeeze(dim=0).size(0),
            ),
            value=0,
        )
        attention_mask = torch.ones_like(tokenized_cd)

        return tokenized_cd, context_mask, attention_mask, full_prompt, tokenized_context
    
    def collate_decoder_fn(self, batch: Any, *args, **kwargs) -> Any:
        idx = [sample["id"] for sample in batch]
        gold_definitions = []
        contexts=[]
        context_texts = []
        gold_titles = []
        batch_candidates = []

        wikidata = []
        title_starts = []
        if self.target == 'title':
            gold_definition=[" " + sample['gold_title'] for sample in batch]
        elif self.target == 'title_def':
            gold_definition=[sample['gold_title_def'] for sample in batch]
        elif self.target == 'definition':
            gold_definition=[sample['gold_definition'] for sample in batch]

        for sample in batch:
            sample_text_context = "[QRY] " + sample["context"] + " [/QRY]"

            if self.mention_extra_contexts > 0:
                mentions = sample['candidates_RETRIEVER'][:self.mention_extra_contexts]
                for mention in mentions:
                    sample_text_context += f"[CTX] {mention['text']} [/CTX]. "

            wikidata.append(sample["wikidata"])
            title_starts.append(sample["gold_title"] + " <def>")
            context_texts.append(sample_text_context)
            contexts.append(sample_text_context)
            gold_titles.append(sample['gold_title'])
            batch_candidates.append(sample['candidates'])
            
            if self.target == 'title':
                gold_target=" "+sample['gold_title']
            elif self.target == 'title_def':
                gold_target=sample['gold_title_def']
            elif self.target == 'definition':
                gold_target=sample['gold_definition']
        
            gold_definitions.append(gold_target)
                
        tokenized_cds   = []
        context_masks   = []
        attention_masks = []
        token_contexts = []
        token_contexts_masks = []
        input_text = []
        candidates = []

        for c, d, t, cand in zip(contexts, gold_definitions, gold_titles, batch_candidates):
            tokenized_cd, context_mask, attention_mask, full_prompt, token_context = self.generate_and_tokenize_prompt(c, d)
            tokenized_cds.append(tokenized_cd)
            token_contexts.append(token_context.input_ids.squeeze(dim=0))
            token_contexts_masks.append(token_context.attention_mask.squeeze(dim=0))
            context_masks.append(context_mask)
            attention_masks.append(attention_mask)
            input_text.append(full_prompt)
            candidates.append(cand)
        
        data = {
            "input_ids": pad_sequence(
                tokenized_cds,
                padding_value=self.tokenizer.pad_token_id,
                batch_first=True,
                padding_side='left'
            ),
            "attention_mask": pad_sequence(
                attention_masks,
                padding_value=0,
                batch_first=True,
                padding_side='left'
            ),
        }

        context_data = {
            "input_ids": pad_sequence(
                token_contexts,
                padding_value=self.tokenizer.pad_token_id,
                batch_first=True,
                padding_side='left'
            ),
            "attention_mask": pad_sequence(
                token_contexts_masks,
                padding_value=0,
                batch_first=True,
                padding_side='left'
            ),
        }

        tokenized_batch, prompt_mask = BatchEncoding(
            data=data,
        ), pad_sequence(context_masks, padding_value=0, batch_first=True), 

        tokenized_context_batch = BatchEncoding(
            data=context_data,
        )

        labels = tokenized_batch.input_ids.clone()
        labels[(prompt_mask == 1) | (tokenized_batch.attention_mask == 0)] = -100

        mask = labels != -100

        batch_inputs = {
            "ids": idx,
            "context_input_ids": tokenized_batch.input_ids,
            "context_attention_mask": tokenized_batch.attention_mask,
            "target_input_ids": labels,
            "target_attention_mask": mask,
            "gold_definition": gold_definition,
            "title_starts": title_starts,
            "wikidata": wikidata,
            "token_context": tokenized_context_batch.input_ids,
            "token_context_mask": tokenized_context_batch.attention_mask,
            "input_text": input_text,
            "candidates": candidates
            }

        return batch_inputs   
