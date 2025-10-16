import argparse
from collections import defaultdict
import json
from pathlib import Path
from tqdm import tqdm

# import ijson


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

    # first count the number of documents for each mention
    mentions = defaultdict(list)
    with input_file.open("r") as f:
        for document in tqdm(f, desc="Counting mentions"):
            document = json.loads(document)
            mention = document["mention"]
            mentions[mention].append(
                document
            )

    # clean up the documents that have the same "window" field
    for mention, documents in mentions.items():
        seen = set()
        new_documents = []
        for d in documents:
            if d["window"] not in seen:
                new_documents.append(d)
                seen.add(d["window"])
        mentions[mention] = new_documents

    with output_file.open("w") as out:
        for mention, documents in mentions.items():
            docs_to_write = []
            # add a filder that removes documents that are made mostly of lists of text
            # a list is usually made of a lot of \n characters

            filtered_documents = []
            for doc in documents:
                if doc["text"].count("\n") / len(doc["text"]) < 0.5:
                    filtered_documents.append(doc)

            if len(documents) < args.minimum:
                docs_to_write = documents
            else:
                docs_to_write = sorted(
                    documents, key=lambda x: x["similarity"], reverse=True
                )[: args.top_k]

            for i, d in enumerate(docs_to_write):
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
