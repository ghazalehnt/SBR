# we want to identify the most informative stuff by idf value.  this script does not care about book description.
# 3 granularity:
# 1- words
# 2- phrases, n-grams
# 3- sentences (accumulate high idf weights).  -> length normalization?
# we would sort the text and create a profile
import argparse
import json
from collections import Counter
from os.path import join, exists

import torch
import numpy as np

from torchtext.data.utils import get_tokenizer, ngrams_iterator

from SBR.utils.data_loading import load_crawled_goodreads_dataset

# todo hard coded params
idf_ngram_year = 'all'
idf_ngram_alpha = True
idf_ngram_path = open('data/paths_vars/GoogleNgram_extracted_IDFs', 'r').read().strip()


def tokenize_function_torchtext(samples, tokenizer=None, doc_desc_field="text", include_ngrams=None,
                                case_sensitive=True, normalize_negation=True, unique=False, do_tf=False):
    if include_ngrams is None:
        raise ValueError("the include_ngrams is not given!")
    n = max(include_ngrams)
    tokenized_batch = []
    tf_ret = []
    for text in samples[doc_desc_field]:
        tokens = tokenizer(text)
        if not case_sensitive:
            tokens = [t.lower() for t in tokens]
        if normalize_negation:
            while "n't" in tokens:
                tokens[tokens.index("n't")] = "not"
        if unique and not do_tf:
            tokens = list(set(tokens))
        phrases = [ng for ng in list(ngrams_iterator(tokens, n)) if len(ng.split()) in include_ngrams]
        if do_tf:
            counter = Counter(phrases)
            phrases = [k for k, v in counter.items()]
            tfs = [v for k, v in counter.items()]
            tf_ret.append(tfs)
        tokenized_batch.append(phrases)

    if do_tf:
        return {f"tokenized_{doc_desc_field}": tokenized_batch, "tf": tf_ret}
    return {f"tokenized_{doc_desc_field}": tokenized_batch}


def tokens_idf_weights(samples, idf_weights=None):
    ret = []
    for tokens in samples['tokenized_text']:
        ret.append([k for k, v in sorted({token: idf_weights[token] if token in idf_weights else 0 for token in tokens}.items(), reverse=True, key=lambda x: x[1])])
    return {f"sorted_tokens": ret}


def tokens_tf_idf_weights(samples, idf_weights=None):
    ret = []
    for tokens, tfs in zip(samples['tokenized_text'], samples['tf']):
        ret.append([k for k, v in sorted({token: idf_weights[token]*tf if token in idf_weights else 0 for token, tf in zip(tokens, tfs)}.items(), reverse=True, key=lambda x: x[1])])
    return {f"sorted_tokens": ret}


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
    print(config['dataset'])
    dataset_config = config['dataset']

    if "<DATA_ROOT_PATH>" in dataset_config["dataset_path"]:
        dataset_config["dataset_path"] = dataset_config["dataset_path"] \
            .replace("<DATA_ROOT_PATH>", open("data/paths_vars/DATA_ROOT_PATH").read().strip())

    datasets, user_info, item_info = load_crawled_goodreads_dataset(dataset_config)

    phrase_sizes = set()  # e.g. [1], [2], [3], [1,2], ..., [1,2,3]
    unique_phrases = False  # if the filtered profile contain only the unique phrases or should they repeat. e.g. t1 t1 t1 t2 t2, ... OR t1 t2 ...
    do_sentence = False
    do_tfidf = False
    # filtering the user profile
    # filter-type1.1 idf_sentence
    if dataset_config['user_text_filter'].startswith("idf_sentence"):
        do_sentence = True
        phrase_sizes.add(1)
    # filter-type1 idf: we can have idf_1_all, idf_2_all, idf_3_all, idf_1-2_all, ..., idf_1-2-3_all, idf_1_unique, ...
    elif dataset_config['user_text_filter'].startswith("idf_"):
        sp = dataset_config['user_text_filter'].split('_')
        for n in sp[1].split('-'):
            phrase_sizes.add(int(n))
        if sp[2] not in ['repeating', 'unique']:
            raise ValueError(f"{dataset_config['user_text_filter']} not implemented or wrong input!")
        if sp[2] == 'unique':
            unique_phrases = True
    # filter-type2 tf-idf: tf-idf_1, ..., tf-idf_1-2-3
    elif dataset_config['user_text_filter'].startswith("tf-idf_"):
        sp = dataset_config['user_text_filter'].split('_')
        for n in sp[1].split('-'):
            phrase_sizes.add(int(n))
        do_tfidf = True
    else:
        raise ValueError(
            f"filtering method not implemented, or belong to another script! {dataset_config['user_text_filter']}")

    # tokenizer = get_tokenizer("basic_english")
    tokenizer = get_tokenizer("spacy")
    user_info = user_info.map(tokenize_function_torchtext, fn_kwargs={
        'tokenizer': tokenizer,
        'case_sensitive': dataset_config['case_sensitive'],
        'normalize_negation': dataset_config['normalize_negation'],
        'unique': unique_phrases,
        'include_ngrams': phrase_sizes,
        'do_tf': do_tfidf
    },
                              batched=True)

    # now load idf weights:
    # do it only for the vocab
    vocab = {n: [] for n in phrase_sizes}
    for tokens in user_info['tokenized_text']:
        for token in tokens:
            if len(token.split()) > 0:
                vocab[len(token.split())].append(token)
    vocab = {n: set(v) for n, v in vocab.items()}

    # load from files
    idf_weights = {}
    for n in phrase_sizes:
        ngram_file = join(idf_ngram_path,
                          f"{n}_gram_casesensitive-{dataset_config['case_sensitive']}_year-{idf_ngram_year}_alphabetic-{idf_ngram_alpha}.json")
        temp = json.load(open(ngram_file, 'r'))
        # filter for the vocab at hand:
        temp = {k: v for k, v in temp.items() if k in vocab[n]}
        idf_weights.update(temp)

    # sentencizer if needed after tokenization is done
    if do_sentence:
        # TODO implement either get it into the tokenizer function or not
        # split_text_into_sentences(
        #     text='This is a paragraph. It contains several sentences. "But why," you ask?',
        #     language='en'
        # )
        # splitter = SentenceSplitter(language='en')
        # splitter.split(text='This is a paragraph. It contains several sentences. "But why," you ask?')
        pass
        # user_sentences = user_info
    else:
        # now let's apply the weights to user data and sort the phrases.
        if do_tfidf:
            user_info = user_info.map(tokens_tf_idf_weights, fn_kwargs={'idf_weights': idf_weights}, batched=True)
        else:
            user_info = user_info.map(tokens_idf_weights, fn_kwargs={'idf_weights': idf_weights}, batched=True)

    if do_tfidf:
        user_info = user_info.remove_columns(['tf'])
    user_info = user_info.remove_columns(['text', 'tokenized_text'])
    user_info = user_info.to_pandas()
    user_info['text'] = user_info['sorted_tokens'].apply(" ".join)
    user_info = user_info.drop(columns=['sorted_tokens'])

    user_info.to_csv(join(dataset_config['dataset_path'],
                          f"users_"
                          f"{'-'.join(config['dataset']['user_text'])}_"
                          f"{config['dataset']['user_review_choice']}_"
                          f"filter-{dataset_config['user_text_filter']}_"
                          f"cs-{dataset_config['case_sensitive']}_"
                          f"nn-{dataset_config['normalize_negation']}.csv"))
    print("done")


# main("configs/example_dataset/precalc.json")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', '-c', type=str, default=None, help='config file, to train')
    args, _ = parser.parse_known_args()

    if not exists(args.config_file):
        raise ValueError(f"Config file does not exist: {args.config_file}")
    main(config_file=args.config_file)
