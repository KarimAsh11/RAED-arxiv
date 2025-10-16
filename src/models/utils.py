import logging
import torch
from functools import wraps
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support, accuracy_score, classification_report
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, T5ForConditionalGeneration
from tokenizers import AddedToken
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model  
from src.models.FiD import FiDT5
import os 

    
def skip_on_OOM(empty_grad=False):
    def _decorator(method):
        @wraps(method)
        def _impl(self, *method_args, **method_kwargs):
            try:
                return method(self, *method_args, **method_kwargs)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    logging.error('OOM error, skipping batch')
                    if empty_grad:
                        for p in self.model.parameters():
                            if p.grad is not None:
                                del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    return
        return _impl
    return _decorator

def add_special_tokens(tokenizer, add_bos=False):
    # special_tokens_dict = {'additional_special_tokens': ["[CTX]","[/CTX]", "[DEF]","[/DEF]", "[QRY]", "[/QRY]", "<def>", "[ENT_MASK]"]}
    special_tokens_dict = {'additional_special_tokens': ["[CTX]","[/CTX]", "[DEF]","[/DEF]", "[QRY]", "[/QRY]", "[ENT_DESC]", "[ENT_TIT]", "[ENT_DEF]"]}

    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict, replace_additional_special_tokens=False)
    if add_bos:
        tokenizer.add_special_tokens({'bos_token': '<|startoftext|>', 'pad_token': '<|padtext|>', 'unk_token': '<|unktext|>'})
    else:
        tokenizer.add_special_tokens({'pad_token': '<|padtext|>', 'unk_token': '<|unktext|>'})
    
    print("Special tokens: ", tokenizer.all_special_tokens)

def prep_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, legacy=True)
    add_special_tokens(tokenizer)
    return tokenizer

def prep_model(model_name):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model    

def prep_model_tokenizer(model_name):
    tokenizer = prep_tokenizer(model_name)
    model = prep_model(model_name)
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model

def prep_fid_model(model_name, tokenizer):
    t5 = T5ForConditionalGeneration.from_pretrained(model_name)
    t5.resize_token_embeddings(len(tokenizer))
    model = FiDT5(t5.config)
    model.load_t5(t5.state_dict())
    return model    

def prep_fid_model_tokenizer(model_name):
    tokenizer = prep_tokenizer(model_name)
    model = prep_fid_model(model_name, tokenizer)
    return tokenizer, model

def prep_smol_tokenizer(model_name):
    # tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, legacy=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_size="left")
    add_special_tokens(tokenizer, add_bos=True)
    return tokenizer

def prep_smol_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print("Loaded model: SmolLM")
    return model  

def prep_smol_model_tokenizer(model_name):
    print("Loading Smol model ...")
    tokenizer = prep_smol_tokenizer(model_name)
    print("Loaded Smol tokenizer ...")
    model = prep_smol_model(model_name)
    print("Loaded Smol model ...")
    model.resize_token_embeddings(len(tokenizer))
    model.generation_config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    print("Resized Smol model ...")
    return tokenizer, model


def prep_llama_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_size="left")
    add_special_tokens(tokenizer, add_bos=False)
    return tokenizer

def prep_llama_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print("Loaded model: Llama-3.2")
    return model  

def prep_llama_model_tokenizer(model_name):
    print("Loading Llama-3.2 model ...")
    tokenizer = prep_llama_tokenizer(model_name)
    print("Loaded Llama-3.2 tokenizer ...")
    model = prep_llama_model(model_name)
    print("Loaded Llama-3.2 model ...")
    model.resize_token_embeddings(len(tokenizer))
    model.generation_config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    print("Resized Llama-3.2 model ...")
    return tokenizer, model


def prep_raed_model_tokenizer(model):
    if "smol" in model.model_name.lower():
        pl_tokenizer, pl_model = prep_smol_model_tokenizer(model.model_name)
    elif model.fid:
        pl_tokenizer, pl_model = prep_fid_model_tokenizer(model.model_name)
    else:   
        pl_tokenizer, pl_model = prep_model_tokenizer(model.model_name)

    return pl_tokenizer, pl_model