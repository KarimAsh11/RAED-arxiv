import argparse
import json
import logging
import sys
# import os
# from pathlib import Path
# import time
# from typing import List, Optional, Union

# import torch
# import tqdm

# from goldenretriever import GoldenRetriever
from goldenretriever.common.log import get_logger
# from goldenretriever.common.model_inputs import ModelInputs
# from goldenretriever.data.base.datasets import BaseDataset

logger = get_logger(level=logging.INFO)


def compute_retriever_stats(dataset, top_k) -> None:
    # correct, total = 0, 0
    correct = []
    for sample in dataset:
        # print(sample)
        window_candidates = sample["candidates_RETRIEVER"][:top_k]
        titles_candidates = [c["id"].rsplit("-")[0] for c in window_candidates]
        titles_candidates = [c.replace(" ", "_").lower() for c in titles_candidates]
        
        correct_contexts = 0
        title_label = sample["wikipedia"].replace(" ", "_").lower()
        # print(f"Title label: {title_label}")
        for title in titles_candidates:
            if title == title_label:
                correct_contexts += 1
            # total += 1
        # print(f"Correct contexts: {correct_contexts}")
        correct.append(correct_contexts / top_k)

        # sys.exit()
    logger.info(f"Total: {len(correct)}")
    # logger.info(f"Correct: {correct}")
    logger.info(f"Corrext / top-k ratio: {sum(correct) / len(correct)}")


def eval(input_path, top_k):
    with open(input_path, "r") as f:
        dataset = [json.loads(line) for line in f]

    compute_retriever_stats(dataset, top_k)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--input_path", type=str, required=True)
    arg_parser.add_argument("--top_k", type=int, default=100)

    eval(**vars(arg_parser.parse_args()))
