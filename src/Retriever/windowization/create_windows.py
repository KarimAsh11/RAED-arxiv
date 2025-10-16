import json

import argparse
from collections import defaultdict
import json
from pathlib import Path
from tqdm import tqdm
import random
import re
import pickle


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("index_file", type=str)
    arg_parser.add_argument("wiki_pages", type=str)
    arg_parser.add_argument("output_file", type=str)
    arg_parser.add_argument("--mapping-file", type=str)
    arg_parser.add_argument("--frequencies", type=str)

    # arg_parser.add_argument("bad_file", type=str)
    args = arg_parser.parse_args()

    index_file = Path(args.index_file)
    wiki_file = Path(args.wiki_pages)
    output_file = Path(args.output_file)

    # bad_file = Path(args.bad_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    bad_cnt = 0
    real_bad_cnt = 0
    really_really_bad = 0

    with open(index_file, "r") as f:
        aida_entities = json.load(f)

    entities_mapping = {}
    if args.mapping_file:
        with open(args.mapping_file) as f:
            entities_mapping = json.load(f)

    # normalize the aida entities
    aida_entities = {k.lower().replace(" ", "_"): v for k, v in aida_entities.items()}

    pages_with_aida_entities = set()
    for entity, pages in aida_entities.items():
        for page in pages:
            pages_with_aida_entities.add(page.lower().replace(" ", "_"))

    print("Pages with aida entities: ", len(pages_with_aida_entities))

    frequencies = {}
    if args.frequencies:
        with open(args.frequencies, "r") as f:
            frequencies = json.load(f)

        # filter the pages_with_aida_entities based on the frequencies
        # order the pages based on the frequencies
        frequencies = {
            k.lower().replace(" ", "_"): v
            for k, v in sorted(
                frequencies.items(), key=lambda item: item[1], reverse=True
            )
        }
        frequencies = set(list(frequencies.keys())[: len(frequencies) // 2])
        # keep only half of the pages
        pages_with_aida_entities_filter = pages_with_aida_entities.intersection(
            frequencies
        )

        pages_with_aida_entities = pages_with_aida_entities_filter
        print(
            "Pages with aida entities after filtering: ", len(pages_with_aida_entities)
        )

    whitespace_regex = re.compile(r"[,\-\[\]\S]+")

    with wiki_file.open("r") as f, output_file.open("w") as f_out:
        for i, page in tqdm(
            enumerate(f), desc="Loading Wiki Pages", total=len(pages_with_aida_entities)
        ):
            try:
                page = json.loads(page)
            except:
                print("Error in loading page")
                continue
            title = page["title"].lower().replace(" ", "_")

            # if title != "louise_currey":
            #     continue

            # if title not in the pages we are interested in, skip
            if title not in pages_with_aida_entities:
                # keep if the title is in the aida entities
                # if title is not in the aida entities, skip
                if title not in aida_entities:
                    continue

            page_text = page["text"]
            for link in page["links"]:
                surface_form = link["label"]
                surface_len = len(surface_form.split())
                mention = link["title"].lower().replace(" ", "_")

                if mention not in aida_entities:
                    continue

                wikidata = link["wikidata"]

                mention_start = link["boundaries"][0]
                mention_end = link["boundaries"][1]
                mention_len = mention_end - mention_start

                span_start = max(0, mention_start - 1000)
                span_end = min(len(page_text), mention_end + 1000)

                span_text = page["text"][span_start:span_end]

                mention_start -= span_start
                mention_end -= span_start

                window_size = 100

                # whitespace_regex = re.compile(r"\S+")
                white_split = {}
                inv_white_split = {}

                for i, m in enumerate(whitespace_regex.finditer(span_text)):
                    white_split[m.start()] = {
                        "token": m.group(),
                        "start_char": m.start(),
                        "end_char": m.end(),
                        "index": i,
                    }
                    # white_split[m.start()] = (i, m.group(0), m.start(), m.end())
                    inv_white_split[i] = m.start()

                # Calculate the range for the start position of the window
                if mention_start not in white_split:
                    really_really_bad += 1
                    continue

                mention_start_index = white_split[mention_start]["index"]
                mention_end_index = (
                    white_split[mention_end]["index"]
                    if mention_end in white_split
                    else mention_start_index
                )

                tokens_before_mention = random.randint(0, window_size - 5)
                tokens_after_mention = window_size - tokens_before_mention

                window_start = max(0, mention_start_index - tokens_before_mention)
                window_end = min(
                    len(white_split) - 1, mention_end_index + tokens_after_mention
                )

                tokenized_window = [
                    white_split[inv_white_split[token_index]]
                    for token_index in range(window_start, window_end + 1)
                ]

                shift = white_split[inv_white_split[window_start]]["start_char"]
                shift_ind = white_split[inv_white_split[window_start]]["index"]

                # window = " ".join([token["token"] for token in tokenized_window])
                window = span_text[
                    tokenized_window[0]["start_char"] : tokenized_window[-1]["end_char"]
                ]

                for token in tokenized_window:
                    token["start_char"] -= shift
                    token["end_char"] -= shift
                    token["index"] -= shift_ind

                span = [
                    mention_start - shift,
                    mention_end - shift,
                ]

                sample = {
                    "mention": mention,
                    "source": title,
                    "window": window,
                    "span": span,
                    "label": surface_form,
                }

                meow1 = sample["window"][sample["span"][0] : sample["span"][1]]
                meow2 = sample["label"]

                if meow1 == meow2:
                    f_out.write(json.dumps(sample) + "\n")

                elif meow1.replace("_", " ") == meow2:
                    bad_cnt += 1
                    # f_bad.write(json.dumps(sample) + "\n")
                else:
                    real_bad_cnt += 1
                    # f_bad.write(json.dumps(sample) + "\n")

    print("Bad count: ", bad_cnt)
    print("Real bad count: ", real_bad_cnt)
    print("Really really bad: ", really_really_bad)
