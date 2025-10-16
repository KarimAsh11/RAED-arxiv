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

class PerplexCallback(Callback):
    def __init__(self, output_path):
        self.output_path = output_path
        self.max_print = 25

    def prompt_prep(self, context):
        full_prompt =f"{context}"
        # full_prompt =f"Generate a title for the mention of the given entity. {context}"
        # full_prompt =f"Generate a definition for the mention of an entity between [DEF] and [/DEF]. ### Context: {context} ### Target Entity Title:"
        # full_prompt = re.sub(r'\n+', ' ', full_prompt)
        return full_prompt
    
    def on_validation_epoch_start(self, trainer, pl_module):
        print("EVALUATING Validation Set ...")
        pl_module.eval()
        perplexities = []
        perp_labels = []
        gold_labels = []

        self.mention_extra_contexts = trainer.datamodule.val_dataset.mention_extra_contexts
        with open('output_preds.jsonl', 'w') as f:

            for k in tqdm(range(len(trainer.datamodule.val_dataset)), desc="Computing Perplexities", total=len(trainer.datamodule.val_dataset)):
                sample_id = trainer.datamodule.val_dataset[k]['id']
                sample = trainer.datamodule.val_dataset[sample_id].copy()
                target = trainer.datamodule.val_dataset.target

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
                    print("sample_id: ", sample_id)
                    print("gold_label: ", gold_label)
                    print("gold_title: ", gold_title)
                    print("titles: ", titles)
                    print("cand_defs: ", cand_defs)
                    # print("sample: ", sample)
                    exit()
                if gold_label not in range(len(cand_defs)):
                # if (gold_label is None) or gold_label not in range(len(cand_defs)):
                    cand_defs = [gold_title] + cand_defs
                    gold_label = 0

                input_texts = sample['context']
                # print("mention_extra_contexts: ", self.mention_extra_contexts)
                if self.mention_extra_contexts > 0 and "candidates_RETRIEVER" in sample:
                    retrieved_contexts = [s['text'] for s in sample['candidates_RETRIEVER'][:self.mention_extra_contexts]]
                else:
                    print("No retrieved contexts")
                    retrieved_contexts = []
                sample_context = ""
                for context_string in retrieved_contexts:
                    sample_context += f" [CTX] {context_string} [/CTX]."
                # sample_context += sample_query
                sample_query = "[QRY] " + sample['context'] + " [/QRY]"
                sample_context = sample_query + sample_context
                sample_context = self.prompt_prep(sample_context)

                input_texts = sample_context

                # pred_perp, batch_pred_cand, pred_idx, batch_perp = self.batch_compute_perplexity(input_texts, cand_defs, pl_module.model, pl_module.tokenizer)
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
                # if (gold_label is None) or gold_label not in range(len(cand_defs)):
                    cand_defs = [gold_title] + cand_defs
                    gold_label = 0

                input_texts = sample['context']
                # print("mention_extra_contexts: ", self.mention_extra_contexts)
                if self.mention_extra_contexts > 0 and "candidates_RETRIEVER" in sample:
                    retrieved_contexts = [s['text'] for s in sample['candidates_RETRIEVER'][:self.mention_extra_contexts]]
                else:
                    print("No retrieved contexts")
                    retrieved_contexts = []
                sample_context = ""
                for context_string in retrieved_contexts:
                    sample_context += f" [CTX] {context_string} [/CTX]."
                # sample_context += sample_query
                sample_query = "[QRY] " + sample['context'] + " [/QRY]"
                sample_context = sample_query + sample_context
                sample_context = self.prompt_prep(sample_context)

                input_texts = sample_context

                # pred_perp, batch_pred_cand, pred_idx, batch_perp = self.batch_compute_perplexity(input_texts, cand_defs, pl_module.model, pl_module.tokenizer)
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

    def batch_compute_perplexity(self, text, candidates, model, tokenizer):
        loss_fct = CrossEntropyLoss(reduction="none")
        perplexities = []
        losses = []
        batch_size=2
        batched_perp = []

        # Tokenize input text
        text = [text] * len(candidates)
        encoded_batch = tokenizer(text, return_tensors='pt')
        labels = tokenizer(candidates, padding=True, truncation=True, max_length=512, return_tensors="pt")

        input_ids  = encoded_batch["input_ids"]
        attn_masks = encoded_batch["attention_mask"]
    
        # labels[labels == 0] = -100
        label_ids  = labels["input_ids"]
        label_ids[label_ids == 0] = -100

        label_attn_masks = labels["attention_mask"]  
        # print(" ******************* no of candidates: ", len(candidates))
        for start_index in tqdm(range(0, len(candidates), batch_size)):
            end_index = min(start_index + batch_size, len(candidates))
            batch_labels = label_ids[start_index:end_index].to(model.device)
            batch_labels_attn_mask = label_attn_masks[start_index:end_index].to(model.device)

            batch_input_ids = input_ids[start_index:end_index].to(model.device)
            batch_attn_masks = attn_masks[start_index:end_index].to(model.device)

            with torch.no_grad():
                output = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attn_masks,
                labels=batch_labels,
                decoder_attention_mask=batch_labels_attn_mask,
            )
            out_logits = output.logits
            loss = output.loss
            # Generate logits
            shift_logits = out_logits[:, :-1, :].contiguous()
            shift_labels = batch_labels[:, 1:].contiguous()
            shift_attention_mask_batch = batch_labels_attn_mask[:, 1:].contiguous()
            ce_loss = loss_fct(shift_logits.transpose(1, 2), shift_labels)
            
            perplexity_batch = torch.exp(
                    (ce_loss * shift_attention_mask_batch).sum(1)
                    / shift_attention_mask_batch.sum(1)
                )

            perplexities.append(torch.exp(loss).item())
            batched_perp += perplexity_batch.tolist()
            losses.append(loss.item())

        argmax_index = np.argmin(perplexities)
        argmin_loss = np.argmin(losses)
        argmin_perp = np.argmin(batched_perp)
        argmax_perp = np.argmax(batched_perp)
        print("argmax_index: ", argmax_index)
        print("argmin_loss: ", argmin_loss)
        print("argmin_perp: ", argmin_perp)
        print("argmax_perp: ", argmax_perp)
        
 
        argmax_perplexity = perplexities[argmax_index]
        argmax_candidate = candidates[argmax_index]

        # Calculate perplexity
        return argmax_perplexity, argmax_candidate, argmax_index, perplexities

    def compute_perplexity(self, text, candidates, model, tokenizer):
        # Tokenize input text
        perplexities = []
        losses = []
        
        with torch.no_grad():
            for i in range(len(candidates)):
                input_ids = tokenizer([text], return_tensors='pt').to(model.device)
                labels = tokenizer([candidates[i]], padding=True, truncation=True, max_length=512, return_tensors="pt").to(model.device)

                # Generate logits
                with torch.no_grad():
                    loss = model(
                    input_ids=input_ids.input_ids,
                    attention_mask=input_ids.attention_mask,
                    labels=labels.input_ids,
                    decoder_attention_mask=labels.attention_mask,
                    ).loss

                perplexity = torch.exp(loss)
                perplexities.append(perplexity.item())
                losses.append(loss.item())
            
            perplexity_batch = torch.exp(
                torch.tensor(losses)
            )
            argmin_loss = np.argmin(losses)
            argmin_index = np.argmin(perplexity_batch).item()
            argmax_perplexity = perplexities[argmin_index]
            argmax_candidate = candidates[argmin_index]

            # Calculate perplexity
            return argmax_perplexity, argmax_candidate, argmin_index, perplexities