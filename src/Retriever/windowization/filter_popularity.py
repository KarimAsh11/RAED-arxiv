import argparse
import json
from pathlib import Path

from tqdm import tqdm

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("input_path", type=str, help="Path to the input file")
    arg_parser.add_argument("output_path", type=str, help="Path to the output file")
    arg_parser.add_argument(
        "frequencies_path", type=str, help="Path to the frequencies file"
    )
    arg_parser.add_argument(
        "--propor", type=float, default=0.5, help="Proportion of the dataset to keep"
    )
    args = arg_parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    frequencies_path = Path(args.frequencies_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # read frequencies
    with frequencies_path.open("r") as f:
        frequencies = json.load(f)

    # order the frequencies
    frequencies = dict(sorted(frequencies.items(), key=lambda x: x[1], reverse=True))
    # keep the most frequent entities
    n = int(args.propor * len(frequencies))
    # care only about the keys
    frequencies = list(frequencies.keys())[:n]
    # normalize
    frequencies = [f.replace(" ", "_").lower() for f in frequencies]
    # unique
    frequencies = set(frequencies)

    filtered = 0
    # filter the data
    with input_path.open("r") as f, output_path.open("w") as f_out:
        for line in tqdm(f):
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                print(line)
                continue
            if "source" not in doc:
                print(doc)
                continue
            source = doc["source"]
            mention = doc["mention"]
            keep = source in frequencies or source == mention
            if keep:
                f_out.write(json.dumps(doc) + "\n")
            else:
                filtered += 1

    print(f"Filtered {filtered} windows")
