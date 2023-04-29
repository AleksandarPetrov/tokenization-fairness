#!/usr/bin/env python

import multiprocessing
import pandas
import os
from collections import defaultdict

from tokenizer_interface import ALL_TOKENIZERS

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

DATASET_PATH = "flores200_dataset"

# get all files in the dataset:
langs = [f.replace(".dev", "") for f in os.listdir(os.path.join(DATASET_PATH, "dev")) if os.path.isfile(f"{DATASET_PATH}/dev/{f}")]
langs.sort()
print(f"Found {len(langs)} languages in the dataset:")
for lang in langs:
    print(f"\t{lang}")

# get the RTL languages (lines in flores_rtl.txt    ):
with open("compute/flores_rtl.txt", 'r') as file:
    rtl_langs = [l.strip() for l in file.read().split('\n')]


def process_one_language(lang, reverse=False):

    #load the data
    with open(f"{DATASET_PATH}/dev/{lang}.dev", 'r') as file:
        data_dev = file.read().split('\n')
    with open(f"{DATASET_PATH}/devtest/{lang}.devtest", 'r') as file:
        data_devtest = file.read().split('\n')
    data = data_dev + data_devtest

    examples = [data[i] for i in [53,140,366,504,703,779,794,871,899,936]]
    data_str = " ".join(data)

    #tokenize the data
    dict_len = {"lang": lang}
    dict_unknown = {"lang": lang}
    examples_tokenized = defaultdict(list)
    for tokenizer in ALL_TOKENIZERS:
        tk = tokenizer()
        print(f"Language {lang}: processing tokenizer {tk.pretty_name}.")
        tokens = tk.encode(data_str)
        dict_len[tk.pretty_name] = len(tokens) 
        dict_unknown[tk.pretty_name] = tk.count_unknown(data_str)

        # process the examples:
        for ex in examples:
            processed_tokens, processed_strs = tk.align_tokens_to_text(tk.encode(ex), reverse=reverse)
            examples_tokenized[tk.pretty_name].append({"text": processed_strs, "tokens": processed_tokens})

    # save the examples:
    os.makedirs("assets/examples", exist_ok=True)
    df=pandas.DataFrame(examples_tokenized)
    df.to_json(f"assets/examples/{lang}.json")

    return dict_len, dict_unknown


process_one_language("jpn_Jpan")