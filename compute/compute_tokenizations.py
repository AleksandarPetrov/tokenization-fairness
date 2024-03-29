#!/usr/bin/env python

import multiprocessing
from typing import Type
import pandas
import os
from collections import defaultdict
import tqdm

from tokenizer_interface import ALL_TOKENIZERS, TokenizerInterface

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
            total_num_tokens = sum([len(t) for t in processed_tokens])
            unknown_count = tk.count_unknown(ex)
            examples_tokenized[tk.pretty_name].append({
                "text": ex, 
                "tokens-text": processed_strs, 
                "tokens": processed_tokens,
                "num_tokens": total_num_tokens,
                "unknown_fraction": unknown_count / total_num_tokens,
                })
            

    # save the examples:
    os.makedirs("assets/examples", exist_ok=True)
    df=pandas.DataFrame(examples_tokenized)
    # print(df)
    df.to_json(f"assets/examples/{lang}.json", force_ascii=False, indent=2)

    return dict_len, dict_unknown


# process the one language not in parallel in order to download the models if needed
_ = process_one_language(langs[0])

# process all languages in parallel
# pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
# processed_dicts_list = pool.map(process_one_language, langs)
# pool.close()
# pool.join()

processed_dicts_list = []
for l in tqdm.tqdm(langs):
    processed_dicts_list.append(process_one_language(l))

# replace language code with language full name
language_map = pandas.read_csv("compute/flores_language_map.csv", index_col=1, skipinitialspace=True)
language_map["Language"] = language_map["Language"].str.strip()

# save the raw tokenization lengths
df = pandas.DataFrame([d[0] for d in processed_dicts_list]).set_index("lang")
df = pandas.merge(df, language_map, left_index=True, right_index=True)
df.set_index("Language", inplace=True)
df.to_csv("assets/tokenization_lengths.csv")

# save the raw numbers of unknown tokens
df_unknown = pandas.DataFrame([d[1] for d in processed_dicts_list]).set_index("lang")
df_unknown = pandas.merge(df_unknown, language_map, left_index=True, right_index=True)
df_unknown.set_index("Language", inplace=True)
df_unknown.to_csv("assets/tokenization_unknown.csv")

# save the fraction of unknown tokens
assert((df.columns == df_unknown.columns).all())
assert((df.index == df_unknown.index).all())
df_unknown_fraction = df_unknown.copy()
for col in df.columns:
    df_unknown_fraction[col] /= df[col]
df_unknown_fraction.to_csv("assets/tokenization_unknown_fraction.csv")

# NaN the rows for which we have too many unknown tokens
THRESHOLD_FOR_TOO_MANY_UNKNOWN = 0.1
df[df_unknown_fraction > THRESHOLD_FOR_TOO_MANY_UNKNOWN] = "–––"
df.to_csv("assets/tokenization_lengths_validated.csv")