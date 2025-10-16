import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from tqdm import tqdm


def process_data(
    input_file: str,
    output_file: str,
    index_file: str,
    mapping_file: str = None,
    window_size: int = None,
):
    input_file = Path(input_file)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    entities_mapping = {}
    if mapping_file:
        with open(mapping_file, "r") as f:
            entities_mapping = json.load(f)

    with open(index_file, "r") as f:
        mentions = defaultdict(list)
        for document in tqdm(f, desc="Counting mentions"):
            document = json.loads(document)
            mention = document["title"]
            if mention in entities_mapping:
                mention = entities_mapping[mention]
            mentions[mention].append(document)

    whitespace_regex = re.compile(r"[,\-\[\]\S]+")
    
    no_positives_ids = []
    no_positive_mentions = set()
    no_candidates_mentions = set()
    with open(input_file, "r") as f, open(output_file, "w") as out:
        for sample in tqdm(f, desc="Processing data"):
            sample = json.loads(sample)
            mention = sample["wikipedia"]
            mention = mention.lower().replace(" ", "_")
            positives = mentions[mention]
            if len(positives) == 0:
                no_positive_mentions.add(mention)

            candidates = sample["candidates_WIKIPEDIA"]
            candidates = [
                c["title"].lower().replace(" ", "_")
                for c in candidates
                if c["title"].lower().replace(" ", "_") != mention
            ]
            negatives = []
            for candidate in candidates:
                if candidate in mentions:
                    negatives.extend(mentions[candidate])
                else:
                    no_candidates_mentions.add(candidate)

            question = sample["context"]
            if window_size:
                # windowize the question and put 

            dpr_sample = {
                "id": sample["id"],
                "question": sample["context"],
                "positive_ctxs": positives,
                "negative_ctxs": negatives,
                "hard_negative_ctxs": [],
            }
            if len(dpr_sample["positive_ctxs"]) == 0:
                no_positives_ids.append(sample["id"])
                continue
            out.write(json.dumps(dpr_sample) + "\n")

    print(f"No positives found for {len(no_positives_ids)} samples")
    print(f"Samples with no positives: {no_positives_ids}")
    for mention in no_positive_mentions:
        print(f"No positive mention found for: {mention}")
    for mention in no_candidates_mentions:
        print(f"No candidate mention found for: {mention}")
    print(f"Total no positive mentions: {len(no_positive_mentions)}")
    print(f"Total no candidate mentions: {len(no_candidates_mentions)}")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("input_file", type=str)
    arg_parser.add_argument("output_file", type=str)
    arg_parser.add_argument("index_file", type=str)
    arg_parser.add_argument("--mapping-file", type=str)
    arg_parser.add_argument("--window-size", type=int, default=None)

    process_data(**vars(arg_parser.parse_args()))
