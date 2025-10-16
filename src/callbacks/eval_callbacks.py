from pytorch_lightning.callbacks import Callback
import torch
from tqdm import tqdm
from src.models.utils import skip_on_OOM
import csv
import random
import evaluate

class EvalCallback(Callback):
    def __init__(self, output_path):
        self.output_path = output_path
        self.max_print = 25

    # def clean_sample(self, sample):
    #     if 'input_ids' in sample: del sample['input_ids']
    #     if 'attention_mask' in sample: del sample['attention_mask']
    #     if 'token_type_ids' in sample: del sample['token_type_ids']
    #     label = sample.pop('label')
    #     sample = {'sample_idx': sample['sample_idx'], **sample}
    #     sample['label'] = label
    #     return sample
        
    def save_samples_with_predictions(self, pl_module, trainer, metrics):
        samples_with_predictions = []
        for idx in tqdm(pl_module._validation_ids, desc="Preparing sample predictions"):
            sample_id = trainer.datamodule.val_dataset[idx]['id']
            prediction = pl_module.val_predictions[idx]
            sample = trainer.datamodule.val_dataset[sample_id].copy()
            sample['prediction'] = prediction
            if 'candidates_WIKIDATA' in sample:
                del sample['candidates_WIKIDATA']
            if 'candidates_WIKI' in sample:
                del sample['candidates_WIKI']
            if 'candidates_RETRIEVER' in sample:
                sample['candidates_RETRIEVER'] = [s['text'] for s in sample['candidates_RETRIEVER'][:3]]
                
            samples_with_predictions.append(sample)

        samples_to_write = samples_with_predictions[:self.max_print]
        samples_with_predictions=samples_with_predictions[self.max_print:]


        self.curr_output_path = "predictions-bleu_"+ '{:.4f}'.format(metrics['bleu']) +'-epoch'+f'_{str(trainer.current_epoch)}.tsv'
        
        with open(self.curr_output_path, 'w', newline='') as w:
            writer = csv.writer(w, delimiter='\t')
            for sample in tqdm(samples_to_write, desc="Writing predictions"):
                for key, value in sample.items():
                    # print(key, value)
                    if key == 'candidates': continue
                    if key == 'candidates_RETRIEVER':
                        w.write(key+'\n')
                        for i, c in enumerate(value):
                            w.write(str(c)+'\n\n')
                    else:
                        w.write(key+'\n'+str(value)+'\n\n')
                w.write('\n\n')
            

    def on_validation_epoch_end(self, trainer, pl_module):
        print("EVALUATING ...")
        # print(pl_module._validation_gold[0])
        # print(pl_module.val_predictions[0])
        
        metrics = self.compute_nlg_metrics(pl_module._validation_gold, pl_module.val_predictions)
        for key, value in metrics.items():
            pl_module.log(key, value, prog_bar=True)

        self.save_samples_with_predictions(pl_module, trainer, metrics)

        pl_module.reset_validation()

    # def on_test_epoch_end(self, trainer, pl_module):
    #     print("TESTING ...")
    #     metrics = compute_metrics_nli_binary(pl_module._test_gold, pl_module.test_predictions)
    #     for key, value in metrics.items():
    #         pl_module.log(key, value, prog_bar=True)

    #     self.save_samples_with_predictions(pl_module, trainer, metrics)

    #     pl_module.reset_test()

    def compute_nlg_metrics(self, gold_references, predictions):
        bleu = evaluate.load("bleu")
        bleu_score = bleu.compute(predictions=predictions, references=gold_references)
        metrics = {'bleu': bleu_score['bleu']}

        return metrics
