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


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, tokens):
        node = self.root
        for token in tokens:
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
        node.is_end_of_word = True

    def get_allowed_tokens(self, prefix_tokens):
        node = self.root
        for token in prefix_tokens:
            if token.item() in node.children:
                node = node.children[token.item()]
            else:
                return []
        return list(node.children.keys())

class ConstrainedPerplexCallback(Callback):
    def __init__(self, output_path):
        self.output_path = output_path
        self.max_print = 25

    def process_batch(self, batch, characters_list):
        results = []

        def create_trie(characters_list):
            trie = Trie()
            for character in characters_list:
                tokens = self.tokenizer(character, add_special_tokens=True).input_ids
                # 0 is the bos token id for T5 (actually pad token)
                tokens = [0] + tokens
                trie.insert(tokens)
            return trie
        self.trie = create_trie(characters_list)

    def prefix_allowed_tokens_fn(self, batch_id, input_ids):
        allowed_tokens = self.trie.get_allowed_tokens(input_ids)
        # if no character starts with the generated text, allow eos token
        if allowed_tokens == []:
            allowed_tokens.append(self.tokenizer.eos_token_id)
            allowed_tokens.append(self.tokenizer.convert_tokens_to_ids("."))
        return allowed_tokens

    def on_validation_epoch_start(self, trainer, pl_module):
        print("EVALUATING Validation Set ...")
        pl_module.eval()
        predictions = []
        gold_labels = []

        self.tokenizer = pl_module.tokenizer
        self.model = pl_module.model

        self.mention_extra_contexts = trainer.datamodule.val_dataset.mention_extra_contexts
        with open('aida_test_constrained_preds.jsonl', 'w') as f:

            for k in tqdm(range(len(trainer.datamodule.val_dataset)), desc="Computing Predictions", total=len(trainer.datamodule.val_dataset)):
                sample_id = trainer.datamodule.val_dataset[k]['id']
                sample = trainer.datamodule.val_dataset[sample_id].copy()
                target = trainer.datamodule.val_dataset.target

                titles = [c["title"].replace(" ", "_").lower() for c in sample["candidates"]]
                # gold_def = sample['wikipedia'].replace("_", " ") + " : " + sample["gold_definition"] 
                gold_def = sample['wikipedia'].replace("_", " ")

                if target == "title":
                    cand_defs = [c['title'].replace("_", " ") for c in sample["candidates"]]
                elif target == 'title_def':
                    cand_defs = [c['title'].replace("_", " ") + " : " + c["text"] for c in sample["candidates"]]
                elif target == "definition":
                    cand_defs = [c["text"] for c in sample["candidates"]]

                gold_title = sample['wikipedia'].replace(" ", "_").lower()

                for i, c in enumerate(sample["candidates"]):
                    if gold_title == titles[i]:
                        gold_label = i
                
                if gold_label not in range(len(cand_defs)):
                    cand_defs = [gold_def] + cand_defs
                    gold_label = 0

                processed_batch = self.process_batch(sample, cand_defs)  

                input_texts = sample['context']
                if len(cand_defs) == 0:
                    print('sample id: ', k)
                    gold_labels.append("0")
                    predictions.append("0")
                    continue

                if self.mention_extra_contexts > 0 and "candidates_RETRIEVER" in sample:
                    retrieved_contexts = [s['text'] for s in sample['candidates_RETRIEVER'][:self.mention_extra_contexts]]
                else:
                    retrieved_contexts = []
                sample_context = ""
                for context_string in retrieved_contexts:
                    sample_context += f"[CTX] {context_string} [/CTX]. "
                sample_query = "[QRY] "+ sample['context']+" [/QRY]"
                sample_context += sample_query

                input_texts = sample_context

                # pred_perp, batch_pred_cand, pred_idx, batch_perp = self.batch_compute_perplexity(input_texts, cand_defs, pl_module.model, pl_module.tokenizer)
                pred = self.compute_prediction(input_texts, cand_defs, pl_module.model, pl_module.tokenizer)

                gold_labels.append(gold_def)
                predictions.append(pred)
                sample['gold'] = gold_def
                sample['constrained_pred'] = pred

                f.write(json.dumps(sample) + '\n')
  
            correct = 0
            for i in range(len(gold_labels)):
                if gold_labels[i] == predictions[i]:
                    correct += 1
            accuracy = correct / len(gold_labels)
            print("Accuracy: ", accuracy)

    def on_test_epoch_start(self, trainer, pl_module):
        print("EVALUATING Test Set...")
        pl_module.eval()
        predictions = []
        gold_labels = []
        self.mention_extra_contexts = trainer.datamodule.test_dataset.mention_extra_contexts
        self.tokenizer = pl_module.tokenizer
        self.model = pl_module.model

        with open('ace2004_test_constrained_preds.jsonl', 'w') as f:

            for k in tqdm(range(len(trainer.datamodule.test_dataset)), desc="Computing Predictions", total=len(trainer.datamodule.test_dataset)):
                sample_id = trainer.datamodule.test_dataset[k]['id']
                sample = trainer.datamodule.test_dataset[sample_id].copy()

                titles = [c["title"].replace(" ", "_").lower() for c in sample["candidates"]]
                # gold_def = sample['wikipedia'].replace("_", " ") + " : " + sample["gold_definition"] 
                gold_def = sample['wikipedia'].replace("_", " ")

                # cand_defs = [c['title'].replace("_", " ") + " : " + c["text"] for c in sample["candidates"]]
                cand_defs = [c['title'].replace("_", " ") for c in sample["candidates"]]
                gold_title = sample['wikipedia'].replace(" ", "_").lower()

                for i, c in enumerate(sample["candidates"]):
                    if gold_title == titles[i]:
                        gold_label = i
                

                if gold_label not in range(len(cand_defs)):
                    cand_defs = [gold_def] + cand_defs
                    gold_label = 0

                processed_batch = self.process_batch(sample, cand_defs)  

                input_texts = sample['context']
                if len(cand_defs) == 0:
                    print('sample id: ', k)
                    gold_labels.append("0")
                    predictions.append("0")
                    continue
                retrieved_contexts = [s['text'] for s in sample['candidates_RETRIEVER'][:self.mention_extra_contexts]]
                sample_context = ""
                for context_string in retrieved_contexts:
                    sample_context += f"[CTX] {context_string} [/CTX]. "
                sample_query = "[QRY] "+ sample['context']+" [/QRY]"
                sample_context += sample_query

                input_texts = sample_context

                pred = self.compute_prediction(input_texts, cand_defs, pl_module.model, pl_module.tokenizer)

                gold_labels.append(gold_def)
                predictions.append(pred)

                sample['gold'] = gold_def
                sample['constrained_pred'] = pred

                f.write(json.dumps(sample) + '\n')

                if len(pred.strip()) == 0: 
                    print("Empty prediction")

            correct = 0
            for i in range(len(gold_labels)):
                if gold_labels[i] == predictions[i]:
                    correct += 1
            accuracy = correct / len(gold_labels)
            print("Accuracy: ", accuracy)

    def compute_prediction(self, text, candidates, model, tokenizer):
        # Tokenize input text
        with torch.no_grad():
            input_ids = tokenizer([text], return_tensors='pt')["input_ids"].to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=20,
                    num_beams=3,
                    prefix_allowed_tokens_fn=self.prefix_allowed_tokens_fn,
                )
                decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

            return decoded_output