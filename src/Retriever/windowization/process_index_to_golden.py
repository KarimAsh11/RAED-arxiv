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
    arg_parser.add_argument("--score", type=float, default=0.0)
    arg_parser.add_argument("--minimum", type=int, default=0)
    arg_parser.add_argument("--top-k", type=int, default=30)
    args = arg_parser.parse_args()

    input_file = Path(args.index_file)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # create a dictionary to hold the top-k documents for each mention
    top_docs = defaultdict(list)

    with input_file.open("r") as f:
        for document in tqdm(f, desc="Processing documents"):
            document = json.loads(document)
            mention = document["mention"]

            if mention not in top_docs:
                top_docs[mention] = []
                heapq.heapify(top_docs[mention])zq

            # use a heap to keep the top-k documents
            # the heap is ordered by the "similarity" parameter
            if len(top_docs[mention]) < args.top_k:
                heapq.heappush(
                    top_docs[mention], (document["similarity"], id(document), document)
                )
            else:
                heapq.heappushpop(
                    top_docs[mention], (document["similarity"], id(document), document)
                )

    # write the top-k documents for each mention to the output file
    with output_file.open("w") as out:
        for mention, docs in tqdm(top_docs.items(), desc="Writing documents"):
            docs = sorted(docs, key=lambda x: x[0], reverse=True)
            for i, (_, _, d) in enumerate(docs):
                dpr_document = {
                    "id": f"{mention}-{i}",
                    "title": d["mention"].strip(),
                    "text": d["window"].strip(),
                }
                # add any other fields in the document as metadata
                metadata = {
                    k: v for k, v in d.items() if k not in ["mention", "window"]
                }
                dpr_document.update({"metadata": metadata})
                out.write(json.dumps(dpr_document) + "\n")
