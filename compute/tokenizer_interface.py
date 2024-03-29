#!/usr/bin/env python
from abc import ABC, abstractmethod, abstractproperty
import os
import requests
import tarfile
from typing import List, Union
from itertools import cycle

import tiktoken
from transformers import AutoTokenizer, LlamaTokenizer

import torch
from seamless_communication.models.unity import load_unity_text_tokenizer

# abstract class for tokenizers inheriting from ABC
class TokenizerInterface(ABC):

    NOT_COMPLETE_SYMBOL_ORD = None

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, text: List[int]) -> str:
        raise NotImplementedError

    @abstractproperty
    def pretty_name(self) -> str:
        raise NotImplementedError
    
    @classmethod
    def format_color(cls, text, color):
        """
        Prints the specified text in the specified color.
        """
        colors = {
            "black": "\u001b[40m",
            "red": "\u001b[41m",
            "green": "\u001b[42m",
            "yellow": "\u001b[43m",
            "blue": "\u001b[44m",
            "magenta": "\u001b[45m",
            "cyan": "\u001b[46m",
            "white": "\u001b[47m",
            "reset": "\033[0m",
        }
        if color not in colors:
            raise ValueError("Invalid color: {}".format(color))
        return colors[color] + text + colors["reset"]

    def print_pretty_tokens(self, tokens: List[int], print_total=False):

        token_words = [self.decode([t]) for t in tokens ]
        colors = ["red", "green", "blue", "magenta", "cyan"]
        
        for t, w, c in zip(tokens, token_words, cycle(colors)):
            print(self.format_color(str(t).ljust(max(len(str(t)), len(w)), '~'), c), end="")

        print("")

        for t, w, c in zip(tokens, token_words, cycle(colors)):
            print(self.format_color(str(w).ljust(max(len(str(t)), len(w)), '~'), c), end="")
        print("")
        
        if print_total:
            print(f"Total {len(tokens)} tokens")
        
    def print_pretty_text(self, text: str, print_total=False):
        tokens = self.encode(text)
        self.print_pretty_tokens(tokens, print_total)

    def print_pretty(self, test_or_tokens: Union[str, List[int]], print_total=False):
        if isinstance(test_or_tokens, str):
            self.print_pretty_text(test_or_tokens, print_total=print_total)
        elif isinstance(test_or_tokens, list):
            self.print_pretty_tokens(test_or_tokens, print_total=print_total)
        else:
            raise ValueError(f"Invalid input type for print_pretty. Must be str or list of ints. Found {type(test_or_tokens)}")

    def align_tokens_to_text(self, tokens, reverse=False):
        processed_tokens = []
        processed_strs = []

        pred = []
        for t in tokens:
            unicode_error = False
            dec = ""

            curr = pred + [t]

            try:
                dec = self.decode(curr)
            except UnicodeDecodeError:
                unicode_error = True

            if (len(dec) > 1) or (len(dec)==1 and ord(dec) != self.NOT_COMPLETE_SYMBOL_ORD) or unicode_error:
                processed_tokens.append(tuple(curr))
                processed_strs.append(dec)
                pred = []
            else:
                pred.append(t)

        if reverse:
            processed_tokens = processed_tokens[::-1]
            processed_strs = processed_strs[::-1]

        return processed_tokens, processed_strs

    def latex_pretty(self, text, font="", reverse=False):
        tokens = self.encode(text)
        processed_tokens = []
        processed_strs = []
        wrapword_command = []

        pred = []
        for t in tokens:
            curr = pred + [t]
            dec = self.decode(curr)
            if (len(dec) > 1) or (len(dec)==1 and ord(dec) != self.NOT_COMPLETE_SYMBOL_ORD):
                processed_tokens.append(tuple(curr))
                processed_strs.append(dec)
                if len(curr) == 1:
                    wrapword_command.append(["wrapword"])
                elif len(curr) == 2:
                    wrapword_command.append(["wrapwordleft", "wrapwordright"])
                else:
                    wrapword_command.append(["wrapwordleft"] + ["wrapwordcenter"] * (len(curr)-2) + ["wrapwordright"])
                pred = []
            else:
                pred.append(t)

        if reverse:
            processed_tokens = processed_tokens[::-1]
            processed_strs = processed_strs[::-1]
            wrapword_command = wrapword_command[::-1]

        prefix = """
        \\begin{center}
        \\begingroup
        \\setlength{\\tabcolsep}{2pt}
        \\renewcommand{\\arraystretch}{0}
        \\begin{tabular}{
        """+ "c" * len(processed_tokens) + "}\n"

        codes = " & ".join(["".join(["\\"+ww+"{"+str(t)+"}" for ww, t in zip(ww_tup, token_tup)]) for ww_tup, token_tup in zip(wrapword_command, processed_tokens)]) + "\\\\\n"
        words = " & ".join(["\\wrapword{"+font + s +"}"  for s in processed_strs]) + "\n"

        suffix = """
        \\end{tabular}
        \\endgroup
        \\end{center}
        """

        return prefix + codes + words + suffix

    @abstractmethod
    def count_unknown(self, text: str) -> int:
        raise NotImplementedError

class UTF32_Tokenizer(TokenizerInterface):

    def encode(self, text: str) -> List[int]:
        encoded = text.encode("utf_32_be")
        encoded = [int.from_bytes(encoded[i*4:(i+1)*4], byteorder="big") for i in range(len(encoded)//4)]
        return encoded
    
    def decode(self, tokens: List[int]) -> str:
        tokens_bytes = [t.to_bytes(length=4, byteorder="big") for t in tokens]
        return b"".join(tokens_bytes).decode("utf_32_be")
    
    @property
    def pretty_name(self) -> str:
        return "UTF-32"
    
    def count_unknown(self, test: str) -> int:
        return 0

class OpenAITokenizer(TokenizerInterface):

    tokenizer=None
    tokenizer_name="NotValid"

    NOT_COMPLETE_SYMBOL_ORD = 65533

    def __init__(self):
        if self.tokenizer is None:
            raise NotImplementedError("OpenAITokenizer must be subclassed!")
        self.encoder = tiktoken.get_encoding(self.tokenizer)

    def encode(self, text: str) -> List[int]:
        return self.encoder.encode(text)
    
    def decode(self, tokens: List[int]) -> str:
        return self.encoder.decode(tokens)

    @property
    def pretty_name(self) -> str:
        return self.tokenizer_name
    
    def count_unknown(self, test: str) -> int:
        return 0

class OpenAI_GPT2(OpenAITokenizer):
    tokenizer = "gpt2"
    tokenizer_name = "GPT-2"

class OpenAI_r50k_base(OpenAITokenizer):
    tokenizer = "r50k_base"
    tokenizer_name = "r50k_base"

class OpenAI_p50k_base(OpenAITokenizer):
    tokenizer = "p50k_base"
    tokenizer_name = "p50k_base"

class OpenAI_p50k_edit(OpenAITokenizer):
    tokenizer = "p50k_edit"
    tokenizer_name = "p50k_edit"

class OpenAI_cl100k_base(OpenAITokenizer):
    tokenizer = "cl100k_base"
    tokenizer_name = "cl100k_base"


class HuggingFaceTokenizer(TokenizerInterface):
    tokenizer = None
    tokenizer_name = "NotValid"
    NOT_COMPLETE_SYMBOL_ORD = 65533
    init_kwargs = {}

    def __init__(self):
        if self.tokenizer is None:
            raise NotImplementedError("HuggingFaceTokenizer must be subclassed!")
        self.encoder = AutoTokenizer.from_pretrained(self.tokenizer, **self.init_kwargs)

    def encode(self, text: str) -> List[int]:
        return self.encoder.convert_tokens_to_ids(self.encoder.tokenize(text))
    
    def decode(self, tokens: List[int]) -> str:
        return self.encoder.decode(tokens)

    @property
    def pretty_name(self) -> str:
        return self.tokenizer_name
    
    def count_unknown(self, text: str) -> int:
        unknown_token = self.encoder.convert_tokens_to_ids([self.encoder.unk_token])[0]
        tokens = self.encode(text)
        tokens_wo_unk = [t for t in tokens if t != unknown_token]
        text_wo_unk = self.decode(tokens_wo_unk)
        return max(0, int(len(tokens)*(len(text)-len(text_wo_unk))/len(text)))


class FacebookAI_XMLR_Base(HuggingFaceTokenizer):
    tokenizer = "xlm-roberta-base"
    tokenizer_name = "XLM-RoBERTa"

class FacebookAI_Roberta_Base(HuggingFaceTokenizer):
    tokenizer = "roberta-base"
    tokenizer_name = "RoBERTa"

class FacebookAI_GottBERT(HuggingFaceTokenizer):
    tokenizer = "uklfr/gottbert-base"
    tokenizer_name = "GottBERT"

class FacebookAI_CamemBERT(HuggingFaceTokenizer):
    tokenizer = "camembert-base"
    tokenizer_name = "CamemBERT"

class FacebookAI_M2M100(HuggingFaceTokenizer):
    tokenizer = "facebook/m2m100_418M"
    tokenizer_name = "M2M100"

class Google_FlanT5(HuggingFaceTokenizer):
    tokenizer = "google/flan-t5-xxl"
    tokenizer_name = "FlanT5"

class Google_mT5(HuggingFaceTokenizer):
    tokenizer = "google/mt5-small"
    tokenizer_name = "mT5"

class Google_CANINE(HuggingFaceTokenizer):
    tokenizer = "google/canine-c"
    tokenizer_name = "CANINE"

class Google_ByT5(HuggingFaceTokenizer):
    tokenizer = "google/byt5-base"
    tokenizer_name = "ByT5"

class FacebookAI_MBart50(HuggingFaceTokenizer):
    tokenizer = "facebook/mbart-large-50"
    tokenizer_name = "MBart50"

class VinAI_PhoBERT(HuggingFaceTokenizer):
    tokenizer = "vinai/phobert-base"
    tokenizer_name = "PhoBERT"

class RoCBert(HuggingFaceTokenizer):
    tokenizer = "weiweishi/roc-bert-base-zh"
    tokenizer_name = "RoCBert"

class BigScience_BLOOM(HuggingFaceTokenizer):
    tokenizer = "bigscience/bloom"
    tokenizer_name = "BLOOM"

class ArabicBERT(HuggingFaceTokenizer):
    tokenizer = "asafaya/bert-base-arabic"
    tokenizer_name = "ArabicBERT"
    
class Google_MuRIL(HuggingFaceTokenizer):
    tokenizer = "google/muril-base-cased"
    tokenizer_name = "MuRIL"

class BERTJapanese(HuggingFaceTokenizer):
    tokenizer = "cl-tohoku/bert-base-japanese"
    tokenizer_name = "BERT Japanese"

class Qwen(HuggingFaceTokenizer):
    tokenizer = "Qwen/Qwen-7B-Chat"
    tokenizer_name = "Qwen"
    init_kwargs = {"trust_remote_code": True}

class LLAMA(HuggingFaceTokenizer):
    tokenizer = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer_name = "LLAMA"

class Glot500(HuggingFaceTokenizer):
    tokenizer = "cis-lmu/glot500-base"
    tokenizer_name = "Glot500"

class FacebookAI_SeamlessM4T(TokenizerInterface):

    pretty_name = "SeamlessM4T"

    def __init__(self) -> None:
        super().__init__()
        
        tokenizer = load_unity_text_tokenizer("seamlessM4T_medium")
        self.tokenizer = tokenizer.create_encoder()
        self.detokenizer = tokenizer.create_decoder()

    def encode(self, text: str) -> List[int]:
        return self.tokenizer(text).tolist()
    
    def decode(self, text: List[int]) -> str:
        return str(self.detokenizer(torch.tensor(text))[0])

    def count_unknown(self, text: str) -> int:
        unknown_sequence = [248059, 1]
        # Convert both lists to string for easier matching
        tokens = self.encode(text)
        str_list = ",".join(map(str, tokens))
        str_sub = ",".join(map(str, unknown_sequence))
        str_list = str_list.replace(str_sub, "").strip(",")
        non_unknown_tokens = [int(item) for item in str_list.split(",") if item]
        text_wo_unk = self.decode(non_unknown_tokens)
        return max(0, int(len(tokens)*(len(text)-len(text_wo_unk))/len(text)))


class FacebookAI_NLLB(HuggingFaceTokenizer):
    tokenizer = "facebook/nllb-200-distilled-600M"
    tokenizer_name = "NLLB"

ALL_TOKENIZERS = [
    Glot500,
    FacebookAI_SeamlessM4T,
    FacebookAI_NLLB,
    Qwen,
    LLAMA,
    OpenAI_GPT2,
    OpenAI_r50k_base,
    OpenAI_p50k_base,
    OpenAI_p50k_edit,
    OpenAI_cl100k_base,
    FacebookAI_Roberta_Base,
    FacebookAI_GottBERT,
    FacebookAI_CamemBERT,
    VinAI_PhoBERT,
    RoCBert,
    FacebookAI_XMLR_Base,
    FacebookAI_M2M100,
    FacebookAI_MBart50,
    Google_mT5,
    Google_FlanT5,
    Google_ByT5,
    Google_CANINE,
    BigScience_BLOOM,
    ArabicBERT,
    Google_MuRIL,
    UTF32_Tokenizer,
    BERTJapanese,
]