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
import re

class PerplexDecoderCallback(Callback):
    def __init__(self, output_val_path, output_test_path, tokenizer):
        self.output_val_path = output_val_path
        self.output_test_path = output_test_path
        self.tokenizer = tokenizer
        self.max_print = 25

    def prompt_prep(self, context):
        full_prompt =f"Generate a title for the mention of an entity between [DEF] and [/DEF]. ### Context: {context}### Target Entity Title:"
        full_prompt = self.tokenizer.bos_token + full_prompt
        return full_prompt
    
    def on_validation_epoch_start(self, trainer, pl_module):
        print("EVALUATING Validation Set ...")
        pl_module.eval()
        perplexities = []
        perp_labels = []
        gold_labels = []

        self.mention_extra_contexts = trainer.datamodule.val_dataset.mention_extra_contexts
        with open('output_val_preds.jsonl', 'w') as f:
            for k in tqdm(range(len(trainer.datamodule.val_dataset)), desc="Computing Perplexities", total=len(trainer.datamodule.val_dataset)):
                sample_id = trainer.datamodule.val_dataset[k]['id']
                sample = trainer.datamodule.val_dataset[sample_id].copy()
                target = trainer.datamodule.val_dataset.target

                titles = [c["title"].replace(" ", "_").lower() for c in sample["candidates"]]
                gold_def = sample['wikipedia'].replace("_", " ") + " <def> " + sample["gold_definition"] 

                if target == "title":
                    cand_defs = [" " + c['title'].replace("_", " ") for c in sample["candidates"]]
                elif target == 'title_def':
                    cand_defs = [c['title'].replace("_", " ") + " <def> " + c["text"] for c in sample["candidates"]]
                elif target == "definition":
                    cand_defs = [c["text"] for c in sample["candidates"]]

                gold_title = sample['wikipedia'].replace(" ", "_").lower()
                gold_title_orig = sample['wikipedia'].replace(" ", "_")

                # Compute perplexity
                if len(cand_defs) == 0:
                    print('sample id: ', sample_id)
                    gold_labels.append(0)
                    perp_labels.append(0)
                    continue

                for i, c in enumerate(sample["candidates"]):
                    if gold_title == titles[i]:
                        gold_label = i
                
                if gold_label == "None":
                    print("Gold label not found")
                    print("sample_id: ", sample_id)
                    print("context: ", sample["context"])
                    print("gold_label: ", gold_label)
                    print("gold_title: ", gold_title)
                    print("titles: ", titles)
                    print("cand_defs: ", cand_defs)
                    exit()

                if gold_label not in range(len(cand_defs)):
                    cand_defs = [gold_title_orig] + cand_defs
                    gold_label = 0

                input_texts = sample['context']
                if self.mention_extra_contexts > 0 and "candidates_RETRIEVER" in sample:
                    retrieved_contexts = [s['text'] for s in sample['candidates_RETRIEVER'][:self.mention_extra_contexts]]
                else:
                    print("No retrieved contexts")
                    retrieved_contexts = []
                
                sample_text_context = "[QRY] " + sample["context"] + " [/QRY]"

                if self.mention_extra_contexts > 0:
                    for mention in retrieved_contexts:
                        sample_text_context += f"[CTX] {mention} [/CTX]. "

                sample_context = self.prompt_prep(sample_text_context)

                input_texts = sample_context

                pred_perp, batch_pred_cand, pred_idx, batch_perp = self.compute_perplexity(input_texts, cand_defs, pl_module.model, pl_module.tokenizer)
                gold_labels.append(gold_label)
                perp_labels.append(pred_idx)
                perplexities.append(pred_perp)

                f.write(json.dumps(sample) + '\n')
  
            correct = 0
            for i in range(len(gold_labels)):
                if gold_labels[i] == perp_labels[i]:
                    correct += 1
            accuracy = correct / len(gold_labels)
            print("Accuracy: ", accuracy)
            exit()

    def on_test_epoch_start(self, trainer, pl_module):
        print("EVALUATING Test Set...")
        pl_module.eval()
        perplexities = []
        perp_labels = []
        gold_labels = []
        self.mention_extra_contexts = trainer.datamodule.test_dataset.mention_extra_contexts

        with open('output_test_preds.jsonl', 'w') as f:

            for k in tqdm(range(len(trainer.datamodule.test_dataset)), desc="Computing Perplexities", total=len(trainer.datamodule.test_dataset)):
                sample_id = trainer.datamodule.test_dataset[k]['id']
                sample = trainer.datamodule.test_dataset[sample_id].copy()
                target = trainer.datamodule.test_dataset.target

                titles = [c["title"].replace(" ", "_").lower() for c in sample["candidates"]]
                gold_def = sample['wikipedia'].replace("_", " ") + " <def> " + sample["gold_definition"] 

                if target == "title":
                    cand_defs = [c['title'].replace("_", " ") for c in sample["candidates"]]
                elif target == 'title_def':
                    cand_defs = [c['title'].replace("_", " ") + " <def> " + c["text"] for c in sample["candidates"]]
                elif target == "definition":
                    cand_defs = [c["text"] for c in sample["candidates"]]

                gold_title = sample['wikipedia'].replace(" ", "_").lower()

                # Compute perplexity
                if len(cand_defs) == 0:
                    print('sample id: ', k)
                    gold_labels.append(0)
                    perp_labels.append(0)
                    continue

                for i, c in enumerate(sample["candidates"]):
                    if gold_title == titles[i]:
                        gold_label = i
                
                if gold_label == "None":
                    print("Gold label not found")
                    print("gold_title: ", gold_title)
                    print("titles: ", titles)
                    print("sample_id: ", sample_id)
                    print("cand_defs: ", cand_defs)
                    print("sample: ", sample)
                    exit()

                if gold_label not in range(len(cand_defs)):
                    cand_defs = [gold_title] + cand_defs
                    gold_label = 0

                input_texts = sample['context']
                if self.mention_extra_contexts > 0 and "candidates_RETRIEVER" in sample:
                    retrieved_contexts = [s['text'] for s in sample['candidates_RETRIEVER'][:self.mention_extra_contexts]]
                else:
                    print("No retrieved contexts")
                    retrieved_contexts = []

                sample_text_context = "[QRY] " + sample["context"] + " [/QRY]"

                if self.mention_extra_contexts > 0:
                    for mention in retrieved_contexts:
                        sample_text_context += f"[CTX] {mention} [/CTX]. "

                sample_context = self.prompt_prep(sample_text_context)

                input_texts = sample_context

                pred_perp, batch_pred_cand, pred_idx, batch_perp = self.compute_perplexity(input_texts, cand_defs, pl_module.model, pl_module.tokenizer)
                gold_labels.append(gold_label)
                perp_labels.append(pred_idx)
                perplexities.append(pred_perp)

                f.write(json.dumps(sample) + '\n')
  
            correct = 0
            for i in range(len(gold_labels)):
                if gold_labels[i] == perp_labels[i]:
                    correct += 1
            accuracy = correct / len(gold_labels)
            print("Accuracy: ", accuracy)

    def dec_tokenize(self, prompt):
        result = self.tokenizer(
                prompt,                 
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,)
        return result
    
    def compute_perplexity(self, text, candidates, model, tokenizer):
        # Tokenize input text
        perplexities = []
        losses = []
        
        with torch.no_grad():
            for i in range(len(candidates)):
                input_context = text
                input_context_ids = tokenizer([input_context], return_tensors='pt', add_special_tokens=True).to(model.device)
                text = text + candidates[i] + tokenizer.eos_token

                input_ids = tokenizer([text], return_tensors='pt', add_special_tokens=True).to(model.device)
                target_ids = input_ids.input_ids.clone()
                prompt_len = input_context_ids.input_ids.shape[1]
                target_ids[:, :prompt_len] = -100

                # Generate logits
                with torch.no_grad():
                    loss = model(
                    input_ids=input_ids.input_ids,
                    attention_mask=input_ids.attention_mask,
                    labels=target_ids,
                    ).loss

                perplexity = torch.exp(loss)
                perplexities.append(perplexity.item())
                losses.append(loss.item())
            
            perplexity_batch = torch.exp(
                torch.tensor(losses)
            )
            argmin_loss = np.argmin(losses)
            argmin_index = np.argmin(perplexity_batch).item()
            argmin_perplexity = perplexities[argmin_index]
            pred_candidate = candidates[argmin_index]

            return argmin_perplexity, pred_candidate, argmin_index, perplexities