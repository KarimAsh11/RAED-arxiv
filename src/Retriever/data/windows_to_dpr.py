import argparse
from collections import defaultdict
import heapq
import json
from pathlib import Path
from tqdm import tqdm

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("index_file", type=str)
    arg_parser.add_argument("output_file", type=str)
    args = arg_parser.parse_args()

    input_file = Path(args.index_file)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # create a dictionary to hold the top-k documents for each mention
    top_docs = defaultdict(list)

    with input_file.open("r") as f:
        for document in tqdm(f, desc="Processing documents"):
            document = json.loads(document)
            mention = document["title"]
            top_docs[mention].append(document)

    # write the top-k documents for each mention to the output file
    with output_file.open("w") as out:
        for mention, docs in tqdm(top_docs.items(), desc="Writing documents"):
            for doc in docs:
                dpr_document = {k: v for k, v in doc.items()}
                # replace text key with question key
                question = dpr_document.pop("text")
                span_start, span_end = dpr_document["metadata"]["span"]
                before_target = question[:span_start]
                target = question[span_start:span_end]
                after_target = question[span_end:]
                question = f"{before_target} <define> {target} </define> {after_target}"
                dpr_document["question"] = question
                dpr_document.update(
                    {
                        "positive_ctxs": docs,
                        "negative_ctxs": [],
                        "hard_negative_ctxs": [],
                    }
                )
                out.write(json.dumps(dpr_document) + "\n")
