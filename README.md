# Tokenization unfairness between languages

This repository holds the code for experiments and the project page for the [Language Model Tokenizers Introduce Unfairness Between Languages](https://arxiv.org/abs/2305.15425) paper.

**Abstract:**


_Recent language models have shown impressive multilingual performance, even when not explicitly trained for it. Despite this, concerns have been raised about the quality of their outputs across different languages. In this paper, we show how disparity in the treatment of different languages arises at the tokenization stage, well before a model is even invoked. The same text translated into different languages can have drastically different tokenization lengths, with differences up to 15 times in some cases. These disparities persist across the 17 tokenizers we evaluate, even if they are intentionally trained for multilingual support. Character-level and byte-level models also exhibit over 4 times the difference in the encoding length for some language pairs. This induces unfair treatment for some language communities in regard to the cost of accessing commercial language services, the processing time and latency, as well as the amount of content that can be provided as context to the models. Therefore, we make the case that we should train future language models using multilingually fair tokenizers._


**Repository Structure:**

The tokenizers are defined in `compute/tokenizer_interface.py` and the computation of the parity tables is done in `compute/compute_tokenizations.py`.

The FLORES-200 dataset is provided.

We also provide the computed table in the `assets` directory:

- `tokenization_unknown_fraction.csv`: What fraction of input characters are mapped to the UNK token for all tokenizers and languages.
- `tokenization_lengths.csv`: The tokenization lengths for all tokenizers and languages.
- `tokenization_lengths_validated.csv`: Tokenization lengths where the language-tokenizer pairs with more than 10% characters mapped to UNK tokens removed.

**Cite as:**

```
@inproceedings{petrov2023token_unfairness,
    title = {Language Model Tokenizers Introduce Unfairness Between Languages},
    author = {Petrov, Aleksandar and La Malfa, Emanuele and H. S. Torr, Philip and Bibi, Adel},    
    booktitle = {Advances in Neural Information Processing Systems},
    url = {https://arxiv.org/abs/2305.15425},
    year = {2023}
}
```
