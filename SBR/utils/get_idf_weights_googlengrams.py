from os import listdir
from os.path import join
import gzip

import numpy as np


def idf(N, df, smooth, prob_idf):
    if smooth and prob_idf:
        raise ValueError("Not implemented")
    elif smooth:
        raise ValueError("Not implemented")
    elif prob_idf:
        raise ValueError("Not implemented")
    else:
        return np.log10(N/df)

# total doc num is in the totalcount... match_count, page_count, volume_count
def acc_num_docs(per_year_stat):
    return acc_df_weights(per_year_stat, field=3)


def acc_df_weights(per_year_stat, field=2):
    #year,total_term_count,document_count
    df = 0
    for st in per_year_stat:
        sp = st.split(',')
        df += int(sp[field])
    return df


def get_idf_weights(ngram_dir, n, keys, idf_smooth, idf_prob):
    idf_weights = {}

    # first read total number of documents from totalcount file:
    total_num_docs = 0
    with open(join(ngram_dir, f'{n}-grams', f'totalcounts-{n}'), 'r') as f:
        for line in f:# only one line
            sp = line.strip().split("\t")
            total_num_docs += acc_num_docs(sp)

    files = listdir(join(ngram_dir, f'{n}-grams'))
    counter = 0
    for f in files:
        if f.startswith('totalcounts') or f == 'download.sh':
            continue
        if counter % 100 == 0:
            print(f"{counter} files parsed.")
        counter += 1
        with gzip.open(join(ngram_dir, f'{n}-grams', f), 'rt') as f:
            for line in f:
                sp = line.strip().split("\t")
                k = sp[0]
                if keys is None or k in keys:
                    df = acc_df_weights(sp[1:])
                    idf_weights[k] = idf(total_num_docs, df, idf_smooth, idf_prob)
    print(f"{counter} files parsed.")
    return idf_weights


#
# idfs = get_idf_weights("PATH/GoogleNgrams/", 1, None, False, False)
# print(idfs)
