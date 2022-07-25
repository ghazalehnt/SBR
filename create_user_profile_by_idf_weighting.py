# we want to identify the most informative stuff by idf value.  this script does not care about book description.
# 3 granularity:
# 1- words
# 2- phrases, n-grams
# 3- sentences (accumulate high idf weights).  -> length normalization?
## 2 methods: idf threshold, or top-k highest
## note: probably need to rescale idf weights as they are very small, e.g. exp(idf) todo exp(idf)=exp(log(N/df))=N/df
### note: how about filtering by tf also? to avoid very rare workds???
import json
from os.path import join

import torch
import numpy as np

from torchtext.data.utils import get_tokenizer, ngrams_iterator

from SBR.utils.data_loading import load_crawled_goodreads_dataset
from SBR.utils.get_idf_weights_googlengrams import get_idf_weights


def tokenize_function_torchtext(samples, tokenizer=None, doc_desc_field="text", n=1):
    tokenized_batch = {}
    tokenized_batch[f"tokenized_{doc_desc_field}"] = [list(ngrams_iterator(tokenizer(text), n)) for text in samples[doc_desc_field]]
    return tokenized_batch


def tokens_idf_weights(samples, idf_weights=None):
    enriched_batch = {}
    enriched_batch['idf'] = [[idf_weights[token] if token in idf_weights else 0 for token in tokens] for tokens in samples['tokenized_text']]
    return enriched_batch


def choose_topk(samples, k=100):
    filtered_batch = {}
    filtered_batch['filtered_text_tokens'] = []
    for tokens, weights in zip(samples['tokenized_text'], samples['idf']):
        ind = np.argpartition(weights, -1 * min(len(weights), k))[-1 * min(len(weights), k):]
        ind = ind[np.argsort(np.array(weights)[ind])][::-1][:len(weights)]
        filtered_batch['filtered_text_tokens'].append(list(np.array(tokens)[ind]))
    return filtered_batch


def main(config_file):
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = json.load(open(config_file, 'r'))
    ### todo these are irrelevant to this script but check later:
    if config['model']['precalc_batch_size'] > 1:
        raise ValueError("There is a bug when the batch size is bigger than one. Users/items with only one chunk"
                         "are producing wrong reps.")

    if "<DATA_ROOT_PATH>" in config["dataset"]["dataset_path"]:
        config["dataset"]["dataset_path"] = config["dataset"]["dataset_path"] \
            .replace("<DATA_ROOT_PATH>", open("data/paths_vars/DATA_ROOT_PATH").read().strip())

    datasets, user_info, item_info = load_crawled_goodreads_dataset(config['dataset'])

    # tokenizer = get_tokenizer("basic_english")
    tokenizer = get_tokenizer("spacy")
    user_info = user_info.map(tokenize_function_torchtext, fn_kwargs={'tokenizer': tokenizer, 'n': config['dataset']['user_text_filter_granularity']}, batched=True)

    # we need the vocab to get only the correspinding ngrams
    vocab = []
    for tokens in user_info['tokenized_text']:
        vocab.extend(tokens)
    vocab = set(vocab)
    # TODO other params?
    idf_weights = get_idf_weights(config['dataset']['googlengram_path'],
                                  config['dataset']['user_text_filter_granularity'],
                                  vocab, False, False)
    user_info = user_info.map(tokens_idf_weights, fn_kwargs={'idf_weights': idf_weights}, batched=True)
    # now filter ... top k highest idf weighted terms?
    # as other tokenizers may act differently, this number is just an approximation of how many terms... better to take it bigger as it is sorted
#    k = (config['dataset']['max_num_chunks_user']+1) * config['dataset']['chunk_size']
    k = 100
    user_info = user_info.map(choose_topk, fn_kwargs={'k': k}, batched=True)
    user_info.to_csv(join(config['dataset']['dataset_path'], f"filtered_user_info_idf_top{k}.csv"))

main("configs/example_dataset/filter_user_profile_idf.json")
