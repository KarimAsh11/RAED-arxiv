import argparse
import json

from tqdm import tqdm


def filter_index(
    train_path: str,
    dev_path: str,
    index_path: str,
    output_path: str,
    test_path: str = None,
):
    # train passages
    train_passages = set()
    with open(train_path, "r") as f:
        for sample in tqdm(f, desc="Reading train data"):
            sample = json.loads(sample)
            for candidate in sample["positive_ctxs"]:
                train_passages.add(candidate["id"])
            for candidate in sample["negative_ctxs"]:
                train_passages.add(candidate["id"])

    # dev passages
    dev_passages = set()
    with open(dev_path, "r") as f:
        for sample in tqdm(f, desc="Reading dev data"):
            sample = json.loads(sample)
            for candidate in sample["positive_ctxs"]:
                dev_passages.add(candidate["id"])
            for candidate in sample["negative_ctxs"]:
                dev_passages.add(candidate["id"])

    test_passages = set()
    if test_path:
        with open(test_path, "r") as f:
            for sample in tqdm(f, desc="Reading test data"):
                sample = json.loads(sample)
                for candidate in sample["positive_ctxs"]:
                    test_passages.add(candidate["id"])
                for candidate in sample["negative_ctxs"]:
                    test_passages.add(candidate["id"])

    all_passages = train_passages.union(dev_passages).union(test_passages)

    with open(output_path, "w") as f_out, open(index_path, "r") as f_index:
        for passage in tqdm(f_index, desc="Filtering index"):
            passage = json.loads(passage)
            if passage["id"] in all_passages:
                f_out.write(json.dumps(passage) + "\n")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("train_path", type=str)
    arg_parser.add_argument("dev_path", type=str)
    arg_parser.add_argument("index_path", type=str)
    arg_parser.add_argument("output_path", type=str)
    arg_parser.add_argument("--test-path", type=str)

    filter_index(**vars(arg_parser.parse_args()))
