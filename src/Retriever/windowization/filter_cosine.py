import json
import os
from pathlib import Path
import subprocess
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class BaseDataset(Dataset):
    def __init__(
        self,
        name: str,
        path: Optional[Union[str, os.PathLike, List[str], List[os.PathLike]]] = None,
        data: Any = None,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        if path is None and data is None:
            raise ValueError("Either `path` or `data` must be provided")
        self.path = path
        self.project_folder = Path(__file__).parent.parent.parent
        self.data = data

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
        paths: Union[str, os.PathLike, List[str], List[os.PathLike]],
        *args,
        **kwargs,
    ) -> Any:
        # load data from single or multiple paths in one single dataset
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch: Any, *args, **kwargs) -> Any:
        raise NotImplementedError


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


# Function to process a batch of data
def process_batch(
    data,
    batch_size=512,
    tokenizer=None,
    model=None,
    descriptions_dict=None,
    max_length=None,
    num_workers=4,
):
    dataloader = DataLoader(
        BaseDataset(name="batch_window", data=data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=lambda x: (
            x,
            tokenizer(
                [_x["window"] for _x in x],
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=max_length or tokenizer.model_max_length,
            ),
        ),
    )

    # Get the NLI score
    augmented_data = []
    with torch.autocast(device_type="cuda", dtype=precision), torch.no_grad():
        for i, batch in tqdm(
            enumerate(dataloader), total=len(dataloader), desc="Processing batch"
        ):
            batch_data, encoding = batch
            encoding = {k: v.to("cuda") for k, v in encoding.items()}
            output = model(**encoding)
            sentence_embeddings = mean_pooling(output, encoding["attention_mask"])
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

            batch_definitions = [
                descriptions_dict[sample["mention"].replace(" ", "_").lower()]
                for sample in batch_data
            ]
            # Transform to probabilities, just the element-wise dot product [batch_size, 382] [batch_size, 382] -> [batch_size, 1]
            probabilities = torch.einsum(
                "ij,ij->i", sentence_embeddings, torch.stack(batch_definitions)
            ).tolist()
            # add the probabilities to the data
            for j, sample in enumerate(batch_data):
                sample["similarity"] = probabilities[j]
                augmented_data.append(sample)
    return augmented_data


if __name__ == "__main__":

    # file = "princess_dataset_v5.jsonl"
    # file = "full_good_miss_why.jsonl"
    # file = "wiki-index-full-2-jun.jsonl"
    file = (
        "/media/data/entity-rag/windows/aida+ood+zelda_contexts_en.jsonl"
        # "/media/hdd1/ric/data/entity-rag/wikipedia/windows/aida+ood_contexts_en.popular20%.jsonl"
    )
    descriptions = "/media/data/entity-rag/entity_descriptions_en_wikipedia.jsonl"

    # Load the dataset and process in batches
    batch_size = 512  # Adjust the batch size according to your GPU/CPU memory
    precision = torch.float16  # torch.float32 or torch.float16 or torch.bfloat16
    num_workers = 4
    max_length = 512
    device = "cuda"
    io_batch_size = 100_000

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model.eval()
    model.to(device)

    # Load the descriptions
    data = []
    # titles = []
    with open(descriptions) as f:
        for line in f:
            doc = json.loads(line)
            data.append(doc)
            # titles.append(doc["text"].replace(" ", "_").lower())

    descriptions_dataloader = DataLoader(
        BaseDataset(name="descriptions", data=data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=lambda x: (
            x,
            tokenizer(
                [_x["metadata"]["definition"] for _x in x],
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=max_length or tokenizer.model_max_length,
            ),
        ),
    )

    descriptions_dict = {}
    with torch.autocast(device_type="cuda", dtype=precision), torch.no_grad():
        for i, batch in tqdm(
            enumerate(descriptions_dataloader), total=len(descriptions_dataloader)
        ):
            doc_batch, model_batch = batch
            model_batch = {k: v.to(device) for k, v in model_batch.items()}
            model_output = model(**model_batch)
            sentence_embeddings = mean_pooling(
                model_output, model_batch["attention_mask"]
            )
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            for j, doc in enumerate(doc_batch):
                descriptions_dict[doc["text"].replace(" ", "_").lower()] = (
                    sentence_embeddings[j]
                )

    batch_data = []

    # faster but not compatible with windows
    num_lines = int(
        subprocess.check_output(["wc", "-l", file]).decode("utf-8").split()[0]
    )

    # num_lines = 87247783

    with open(file) as f, open(
        file.replace(".jsonl", "_similarity.jsonl"), "w"
    ) as f_out:
        for idx, line in tqdm(enumerate(f), total=num_lines, desc="Processing"):
            # if 6359929 > idx:
            #     continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(line)
                continue

            if data["mention"].replace(" ", "_").lower() not in descriptions_dict:
                continue
            batch_data.append(data)

            # Process the batch if it reaches the batch size
            if len(batch_data) == io_batch_size:
                batch_data_with_score = process_batch(
                    batch_data,
                    batch_size,
                    tokenizer,
                    model,
                    descriptions_dict,
                    max_length,
                    num_workers,
                )
                for d in batch_data_with_score:
                    f_out.write(json.dumps(d) + "\n")
                batch_data = []

        # Process the remaining data
        if batch_data:
            batch_data_with_score = process_batch(
                batch_data,
                batch_size,
                tokenizer,
                model,
                descriptions_dict,
                max_length,
                num_workers,
            )
            for d in batch_data_with_score:
                f_out.write(json.dumps(d) + "\n")
