import argparse
import json
import os
from collections import defaultdict
from os.path import join
import math

shorten_names = {
    "item.title": "it",
    "item.genres": "ig",
    "item.description": "id",
    "interaction.review_text": "ir",
    "interaction.reviewText": "ir",
    "interaction.summary": "is",

}

shorten_strategies = {
    "item_sentence_SBERT_iitem.title-item.genres-item.description": "SBERTFULL",
    "item_sentence_SBERT_iitem.title-item.genres": "SBERTBASIC",
    "item_sentence_SBERT_iitem.title-item.category-item.description": "SBERTFULL",
    "item_sentence_SBERT_iitem.title-item.category": "SBERTBASIC",
    "random_sentence": "srand",
    "idf_sentence": "sidf",
    "csTrue": "csT",
    "nnTrue": "nnT"
}

metric_name_map = {
    "ndcg_cut_5": "NDCG@5",
    "ndcg_cut_10": "NDCG@10",
    "recip_rank": "MRR",
    "auc": "AUC",
    "P_1": "P@1",
    "P_5": "P@5",
    "micro_ndcg_cut_5": "Micro NDCG@5",
    "micro_ndcg_cut_10": "Micro NDCG@10",
    "micro_recip_rank": "Micro MRR",
    "micro_auc": "Micro AUC",
    "micro_P_1": "Micro P@1",
    "micro_P_5": "Micro P@5",
    "macro_ndcg_cut_5": "Macro NDCG@5",
    "macro_ndcg_cut_10": "Macro NDCG@10",
    "macro_recip_rank": "Macro MRR",
    "macro_auc": "Macro AUC",
    "macro_P_1": "Macro P@1",
    "macro_P_5": "Macro P@5",
}

name_mapping = {
    "MF-200-xavier_normal": "CF",

    "VanillaBERT_ffn_endtoend-200-200-200-200-is-ir_srand_csT_nnT_it-item.category-id_csT_nnT": "CUP$_{rand}$",
    "VanillaBERT_ffn_endtoend-200-200-200-200-is-ir_sidf_csT_nnT_it-item.category-id_csT_nnT": "CUP$_{idf}$",
    "VanillaBERT_ffn_endtoend-200-200-200-200-is-ir_SBERTFULL_csT_nnT_it-item.category-id_csT_nnT": "CUP$_{sbert}$",
    "VanillaBERT_ffn_endtoend-200-200-200-200-is-ir_tf-idf_1_csFalse_nnT_it-item.category-id_csT_nnT": "CUP$_{tfidf-1gram}$",
    "VanillaBERT_ffn_endtoend-200-200-200-200-is-ir_tf-idf_3_csFalse_nnT_it-item.category-id_csT_nnT": "CUP$_{tfidf-3gram}$",
    "VanillaBERT_ffn_endtoend-200-200-200-200-interaction.model_keywords_sr_srand_csT_nnT_it-item.category-id_csT_nnT": "CUP$_{keywords}$",
    "VanillaBERT_ffn_endtoend-200-200-200-200-is-ir_chatgpt_it-item.category-id_csT_nnT": "CUP$_{GPT}$",
    "VanillaBERT_ffn_endtoend-200-200-200-200-is-ir_srand_vocab_it-item.category-id_csT_nnT": "CUP$_{rand-voc}$",
    "VanillaBERT_ffn_endtoend-200-200-200-200-is-ir_sidf_vocab_it-item.category-id_csT_nnT": "CUP$_{idf-voc}$",
    "VanillaBERT_ffn_endtoend-200-200-200-200-is-ir_SBERTFULL_vocab_it-item.category-id_csT_nnT": "CUP$_{sbert-voc}$",
    "VanillaBERT_ffn_endtoend-200-200-200-200-is-ir_tf-idf_1_vocab_it-item.category-id_csT_nnT": "CUP$_{tfidf-1gram-voc}$",
    "VanillaBERT_ffn_endtoend-200-200-200-200-is-ir_tf-idf_3_vocab_it-item.category-id_csT_nnT": "CUP$_{tfidf-3gram-voc}$",
    
    "VanillaBERT_ffn_endtoend-200-200-200-200-ir_srand_csT_nnT_it-ig-id_csT_nnT": "CUP$_{rand}$",
    "VanillaBERT_ffn_endtoend-200-200-200-200-ir_sidf_csT_nnT_it-ig-id_csT_nnT": "CUP$_{idf}$",
    "VanillaBERT_ffn_endtoend-200-200-200-200-ir_SBERTFULL_csT_nnT_it-ig-id_csT_nnT": "CUP$_{sbert}$",
    "VanillaBERT_ffn_endtoend-200-200-200-200-ir_tf-idf_1_csFalse_nnT_it-ig-id_csT_nnT": "CUP$_{tfidf-1gram}$",
    "VanillaBERT_ffn_endtoend-200-200-200-200-ir_tf-idf_3_csFalse_nnT_it-ig-id_csT_nnT": "CUP$_{tfidf-3gram}$",
    "VanillaBERT_ffn_endtoend-200-200-200-200-interaction.model_keywords_r_srand_csT_nnT_it-ig-id_csT_nnT": "CUP$_{keywords}$",
    "VanillaBERT_ffn_endtoend-200-200-200-200-ir_chatgpt_it-ig-id_csT_nnT": "CUP$_{GPT}$",
    "VanillaBERT_ffn_endtoend-200-200-200-200-ir_srand_vocab_it-ig-id_csT_nnT": "CUP$_{rand-voc}$",
    "VanillaBERT_ffn_endtoend-200-200-200-200-ir_sidf_vocab_it-ig-id_csT_nnT": "CUP$_{idf-voc}$",
    "VanillaBERT_ffn_endtoend-200-200-200-200-ir_SBERTFULL_vocab_it-ig-id_csT_nnT": "CUP$_{sbert-rand}$",
    "VanillaBERT_ffn_endtoend-200-200-200-200-ir_tf-idf_1_vocab_it-ig-id_csT_nnT": "CUP$_{tfidf-1gram-voc}$",
    "VanillaBERT_ffn_endtoend-200-200-200-200-ir_tf-idf_3_vocab_it-ig-id_csT_nnT": "CUP$_{tfidf-3gram-voc}$",
}

def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier + 0.5) / multiplier


def wrap_the_num(num, i, group, group_inds):
    if group_inds[group][i]:
        return f"\\textbf{{{num}}}"
    else:
        return num


def print_res():
    for rk in ["uniform", "weighted"]:
        if rk == "uniform":
            print("&\multicolumn{8}{c}{Uniform Negative Training Samples} \\\\")
        else:
            print("&\multicolumn{8}{c}{Weighted Negative Training Samples} \\\\")
        print("&\multicolumn{4}{c}{item:full, user:review} &\multicolumn{4}{c}{+CF} \\\\")
        if rk in res and rk in print_list_col1:
            ###
            inds_per_group1 = dict()
            for g in grps:
                nums_per_group = [round_half_up(res[rk][print_list_col1[rk][i]][g] * 100, 2) if print_list_col1[rk][i] in res[rk] else -1 for i in range(len(print_list_col1[rk]))]
                max_per_group = max(nums_per_group)
                inds_per_group1[g] = [x == max_per_group for x in nums_per_group]
            inds_per_group2 = dict()
            for g in grps:
                nums_per_group = [round_half_up(res[rk][print_list_col2[rk][i]][g] * 100, 2) if print_list_col2[rk][i] in res[rk] else -1 for i in range(len(print_list_col1[rk]))]
                max_per_group = max(nums_per_group)
                inds_per_group2[g] = [x == max_per_group for x in nums_per_group]
            ###
            for i in range(len(print_list_col1[rk])):
                k1 = print_list_col1[rk][i]
                k2 = print_list_col2[rk][i]
                if k1 != "" and k1[:k1.rindex(',')] in name_mapping:
                    p = f"{name_mapping[k1[:k1.rindex(',')]]} & "
                elif k2 != "" and k2[:k2.rindex(',')] in name_mapping:
                    p = f"{name_mapping[k2[:k2.rindex(',')]]} & "
                else:
                    p = ""
                if k1 in res[rk]:
                    p += f"{' & '.join(wrap_the_num(str(round_half_up(res[rk][k1][g] * 100, 2)), i, g, inds_per_group1) for g in grps)} & "
                else:
                    p += f"{''.join(len(grps) * ' & ')} "
                if k2 in res[rk]:
                    p += f"{' & '.join(wrap_the_num(str(round_half_up(res[rk][k2][g] * 100, 2)), i, g, inds_per_group2) for g in grps)}"
                else:
                    p += f"{''.join((len(grps) - 1) * ' & ')} "

                p += f"\\\\ \hline % {k1} {k2} "
                print(p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # required: path to gt and pd to be evaluated:
    parser.add_argument('--dir', '-d', type=str, default=None, help='path to exp dirs')
    parser.add_argument('--metric', '-m', type=str, default=None, help='metric')
    parser.add_argument('--group', '-g', type=str, default=None, help='group')
    parser.add_argument('--resfile', '-r', type=str, default="results_test_neg_standard_100_best_model.txt", help='group')
    args, _ = parser.parse_known_args()

    gr = args.group
    m = args.metric
    dir=args.dir
    resfile = args.resfile
    if 'standard' in resfile:
        ng = "Standard"
    elif 'genres' in resfile:
        ng = "Profile-based"
    elif 'SB_BM25' in resfile:
        ng = "Search-based"
    else:
        raise ValueError("resfile which neg strategy ?")
    grps = ["ALL", "1-5", "6-50", "51+"]
    if "ratios" in resfile:
        grps = ["ALL", "sporadic", "regular", "bibliophilic"]
    dirs = os.listdir(dir)
    res = {"uniform": defaultdict(), "weighted": defaultdict()}
    for d in dirs:
        if os.path.exists(join(dir, d, "config.json")) and os.path.exists(join(dir, d, resfile)):
            config = json.load(open(join(dir, d, "config.json"), 'r'))
            res_key = "uniform"
            if 'random_w_CF' in config['dataset']['training_neg_sampling_strategy']:
                res_key = "weighted"
           
            # only get the runs with bs=256
            if config['dataset']['train_batch_size'] != 256:
                continue

            if config['model']['name'].startswith("MF_ffn"):
                continue
            elif config['model']['name'].startswith("MF"):
                n = f"{config['model']['name']}-{config['model']['embedding_dim']}-{config['model']['embed_init']}"
            elif config['model']['name'].startswith("VanillaBERT_ffn_endtoend"):
                temp1 = config['dataset']['user_text_file_name']
                for s, v in shorten_strategies.items():
                    temp1 = temp1.replace(s, v)
                for s, v in shorten_names.items():
                    temp1 = temp1.replace(s, v)
                temp2 = config['dataset']['item_text_file_name']
                for s, v in shorten_strategies.items():
                    temp2 = temp2.replace(s, v)
                for s, v in shorten_names.items():
                    temp2 = temp2.replace(s, v)
                temp3 = '-'.join([str(s) for s in config['model']['user_k']])
                temp4 = '-'.join([str(s) for s in config['model']['item_k']])
                n = f"{config['model']['name']}-{temp3}-{temp4}-" \
                    f"{temp1}_{temp2}"
                if 'append_embedding_ffn' in config['model'] and config['model']['append_embedding_ffn'] == True:
                    n = f"{config['model']['name']}-emb-{config['model']['user_embedding']}-{config['model']['item_embedding']}-" \
                    f"{temp3}-{temp4}-{temp1}_{temp2}"
                elif 'append_embedding_after_ffn' in config['model'] and config['model']['append_embedding_after_ffn'] == True:
                    n = f"{config['model']['name']}-embafter-{config['model']['user_embedding']}-{config['model']['item_embedding']}-" \
                    f"{temp3}-{temp4}-{temp1}_{temp2}"
            else:
                print(d)
                print(config['model']['name'])
                exit()
            if gr is None:
                g_res = defaultdict()
            for line in open(join(dir, d, resfile), 'r'):
                if len(line.strip()) > 0:
                    r = json.loads(line.replace("\n", "").strip())
                    for g in grps:
                        if g in r:
                            g_res[g] = r[g][m]

            if gr is None:
                res[res_key][f"{n},{config['trainer']['lr']}"] = g_res
            else:
                res[res_key][f"{n},{config['trainer']['lr']}"] = g_res[gr]

    # print(res.keys())
    print("\\begin{table*}[tbh]")
    print_list_col1 = {}
    print_list_col2 = {}
    if "Amazon" in dir:
        print(f"\\caption{{ {metric_name_map[m]} Amazon text rich dataset. {ng} Evaluation. }}")
        print("\\begin{tabular}{|l|l|l|l|l||l|l|l|l|}")
        print("Method & ALL & Sporadic & Regular & Bibliophilic & ALL & Sporadic & Regular & Bibliophilic \\\\ \hline")
        print_list_col1["uniform"] = ["",
                                      "VanillaBERT_ffn_endtoend-200-200-200-200-is-ir_srand_csT_nnT_it-item.category-id_csT_nnT,0.0004",
                                      "VanillaBERT_ffn_endtoend-200-200-200-200-is-ir_sidf_csT_nnT_it-item.category-id_csT_nnT,0.0004",
                                      "VanillaBERT_ffn_endtoend-200-200-200-200-is-ir_SBERTFULL_csT_nnT_it-item.category-id_csT_nnT,0.0004",
                                      "VanillaBERT_ffn_endtoend-200-200-200-200-is-ir_tf-idf_1_csFalse_nnT_it-item.category-id_csT_nnT,4e-05",
                                      "VanillaBERT_ffn_endtoend-200-200-200-200-is-ir_tf-idf_3_csFalse_nnT_it-item.category-id_csT_nnT,4e-05",
                                      "VanillaBERT_ffn_endtoend-200-200-200-200-interaction.model_keywords_sr_srand_csT_nnT_it-item.category-id_csT_nnT,0.0004",
                                      "VanillaBERT_ffn_endtoend-200-200-200-200-is-ir_chatgpt_it-item.category-id_csT_nnT,0.0004",
                                      "VanillaBERT_ffn_endtoend-200-200-200-200-is-ir_srand_vocab_it-item.category-id_csT_nnT,4e-05",
                                      "VanillaBERT_ffn_endtoend-200-200-200-200-is-ir_sidf_vocab_it-item.category-id_csT_nnT,0.0004",
                                      "VanillaBERT_ffn_endtoend-200-200-200-200-is-ir_SBERTFULL_vocab_it-item.category-id_csT_nnT,0.0004",
                                      "VanillaBERT_ffn_endtoend-200-200-200-200-is-ir_tf-idf_1_vocab_it-item.category-id_csT_nnT,0.0004",
                                      "VanillaBERT_ffn_endtoend-200-200-200-200-is-ir_tf-idf_3_vocab_it-item.category-id_csT_nnT,4e-05"]
        print_list_col2["uniform"] = ["MF-200-xavier_normal,4e-05",
                                      "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-is-ir_srand_csT_nnT_it-item.category-id_csT_nnT,0.0004",
                                      "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-is-ir_sidf_csT_nnT_it-item.category-id_csT_nnT,0.0004",
                                      "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-is-ir_SBERTFULL_csT_nnT_it-item.category-id_csT_nnT,0.0004",
                                      "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-is-ir_tf-idf_1_csFalse_nnT_it-item.category-id_csT_nnT,4e-05",
                                      "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-is-ir_tf-idf_3_csFalse_nnT_it-item.category-id_csT_nnT,4e-05",
                                      "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-interaction.model_keywords_sr_srand_csT_nnT_it-item.category-id_csT_nnT,0.0004",
                                      "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-is-ir_chatgpt_it-item.category-id_csT_nnT,0.0004",
                                      "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-is-ir_srand_vocab_it-item.category-id_csT_nnT,0.0004",
                                      "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-is-ir_sidf_vocab_it-item.category-id_csT_nnT,0.0004",
                                      "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-is-ir_SBERTFULL_vocab_it-item.category-id_csT_nnT,0.0004",
                                      "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-is-ir_tf-idf_1_vocab_it-item.category-id_csT_nnT,4e-05",
                                      "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-is-ir_tf-idf_3_vocab_it-item.category-id_csT_nnT,4e-05"]
        print_list_col1["weighted"] = ["",
                                      "VanillaBERT_ffn_endtoend-200-200-200-200-is-ir_srand_csT_nnT_it-item.category-id_csT_nnT,4e-05",
                                      "VanillaBERT_ffn_endtoend-200-200-200-200-is-ir_sidf_csT_nnT_it-item.category-id_csT_nnT,4e-05",
                                      "VanillaBERT_ffn_endtoend-200-200-200-200-is-ir_SBERTFULL_csT_nnT_it-item.category-id_csT_nnT,4e-05",
                                      "VanillaBERT_ffn_endtoend-200-200-200-200-is-ir_tf-idf_1_csFalse_nnT_it-item.category-id_csT_nnT,4e-05",
                                      "VanillaBERT_ffn_endtoend-200-200-200-200-is-ir_tf-idf_3_csFalse_nnT_it-item.category-id_csT_nnT,4e-05",
                                      "VanillaBERT_ffn_endtoend-200-200-200-200-interaction.model_keywords_sr_srand_csT_nnT_it-item.category-id_csT_nnT,4e-05",
                                      "VanillaBERT_ffn_endtoend-200-200-200-200-is-ir_chatgpt_it-item.category-id_csT_nnT,4e-05",
                                      "VanillaBERT_ffn_endtoend-200-200-200-200-is-ir_srand_vocab_it-item.category-id_csT_nnT,4e-05",
                                      "VanillaBERT_ffn_endtoend-200-200-200-200-is-ir_sidf_vocab_it-item.category-id_csT_nnT,4e-05",
                                      "VanillaBERT_ffn_endtoend-200-200-200-200-is-ir_SBERTFULL_vocab_it-item.category-id_csT_nnT,4e-05",
                                      "VanillaBERT_ffn_endtoend-200-200-200-200-is-ir_tf-idf_1_vocab_it-item.category-id_csT_nnT,4e-05",
                                      "VanillaBERT_ffn_endtoend-200-200-200-200-is-ir_tf-idf_3_vocab_it-item.category-id_csT_nnT,4e-05"]
        print_list_col2["weighted"] = ["MF-200-xavier_normal,0.04",
                                      "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-is-ir_srand_csT_nnT_it-item.category-id_csT_nnT,4e-05",
                                      "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-is-ir_sidf_csT_nnT_it-item.category-id_csT_nnT,4e-05",
                                      "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-is-ir_SBERTFULL_csT_nnT_it-item.category-id_csT_nnT,4e-05",
                                      "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-is-ir_tf-idf_1_csFalse_nnT_it-item.category-id_csT_nnT,4e-05",
                                      "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-is-ir_tf-idf_3_csFalse_nnT_it-item.category-id_csT_nnT,4e-05",
                                      "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-interaction.model_keywords_sr_srand_csT_nnT_it-item.category-id_csT_nnT,4e-05",
                                      "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-is-ir_chatgpt_it-item.category-id_csT_nnT,4e-05",
                                      "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-is-ir_srand_vocab_it-item.category-id_csT_nnT,4e-05",
                                      "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-is-ir_sidf_vocab_it-item.category-id_csT_nnT,4e-05",
                                      "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-is-ir_SBERTFULL_vocab_it-item.category-id_csT_nnT,4e-05",
                                      "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-is-ir_tf-idf_1_vocab_it-item.category-id_csT_nnT,4e-05",
                                      "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-is-ir_tf-idf_3_vocab_it-item.category-id_csT_nnT,4e-05"]

        print_res()
    else:
        print(f"\\caption{{ {metric_name_map[m]} Goodreads text rich dataset. {ng} Evaluation. }}")
        print("\\begin{tabular}{|l|l|l|l|l||l|l|l|l|}")
        print("Method & ALL & Sporadic & Regular & Bibliophilic & ALL & Sporadic & Regular & Bibliophilic \\\\ \hline")
        print_list_col1["uniform"] = ["",
                                      "VanillaBERT_ffn_endtoend-200-200-200-200-ir_srand_csT_nnT_it-ig-id_csT_nnT,0.0004",
                                      "VanillaBERT_ffn_endtoend-200-200-200-200-ir_sidf_csT_nnT_it-ig-id_csT_nnT,0.0004",
                                      "VanillaBERT_ffn_endtoend-200-200-200-200-ir_SBERTFULL_csT_nnT_it-ig-id_csT_nnT,0.0004",
                                      "VanillaBERT_ffn_endtoend-200-200-200-200-ir_tf-idf_1_csFalse_nnT_it-ig-id_csT_nnT,4e-05",
                                      "VanillaBERT_ffn_endtoend-200-200-200-200-ir_tf-idf_3_csFalse_nnT_it-ig-id_csT_nnT,0.0004",
                                      "VanillaBERT_ffn_endtoend-200-200-200-200-interaction.model_keywords_r_srand_csT_nnT_it-ig-id_csT_nnT,0.0004",
                                      "VanillaBERT_ffn_endtoend-200-200-200-200-ir_chatgpt_it-ig-id_csT_nnT,0.0004",
                                      "VanillaBERT_ffn_endtoend-200-200-200-200-ir_srand_vocab_it-ig-id_csT_nnT,0.0004",
                                      "VanillaBERT_ffn_endtoend-200-200-200-200-ir_sidf_vocab_it-ig-id_csT_nnT,0.0004",
                                      "VanillaBERT_ffn_endtoend-200-200-200-200-ir_SBERTFULL_vocab_it-ig-id_csT_nnT,0.0004",
                                      "VanillaBERT_ffn_endtoend-200-200-200-200-ir_tf-idf_1_vocab_it-ig-id_csT_nnT,4e-05",
                                      "VanillaBERT_ffn_endtoend-200-200-200-200-ir_tf-idf_3_vocab_it-ig-id_csT_nnT,0.0004"]
        print_list_col2["uniform"] = ["MF-200-xavier_normal,0.0004",
                                      "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-ir_srand_csT_nnT_it-ig-id_csT_nnT,0.0004",
                                      "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-ir_sidf_csT_nnT_it-ig-id_csT_nnT,0.0004",
                                      "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-ir_SBERTFULL_csT_nnT_it-ig-id_csT_nnT,0.0004",
                                      "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-ir_tf-idf_1_csFalse_nnT_it-ig-id_csT_nnT,0.0004",
                                      "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-ir_tf-idf_3_csFalse_nnT_it-ig-id_csT_nnT,0.0004",
                                      "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-interaction.model_keywords_r_srand_csT_nnT_it-ig-id_csT_nnT,0.0004",
                                      "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-ir_chatgpt_it-ig-id_csT_nnT,0.0004",
                                      "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-ir_srand_vocab_it-ig-id_csT_nnT,0.0004",
                                      "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-ir_sidf_vocab_it-ig-id_csT_nnT,0.0004",
                                      "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-ir_SBERTFULL_vocab_it-ig-id_csT_nnT,0.0004",
                                      "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-ir_tf-idf_1_vocab_it-ig-id_csT_nnT,0.0004",
                                      "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-ir_tf-idf_3_vocab_it-ig-id_csT_nnT,0.0004"]
        print_list_col1["weighted"] = ["",
                                       # "VanillaBERT_ffn_endtoend-200-200-200-200-ir_srand_csT_nnT_it-ig-id_csT_nnT,"
                                       "VanillaBERT_ffn_endtoend-200-200-200-200-ir_sidf_csT_nnT_it-ig-id_csT_nnT,4e-05",
                                       "VanillaBERT_ffn_endtoend-200-200-200-200-ir_SBERTFULL_csT_nnT_it-ig-id_csT_nnT,4e-05",
                                       # "VanillaBERT_ffn_endtoend-200-200-200-200-ir_tf-idf_1_csFalse_nnT_it-ig-id_csT_nnT,"
                                       "VanillaBERT_ffn_endtoend-200-200-200-200-ir_tf-idf_3_csFalse_nnT_it-ig-id_csT_nnT,4e-05",
                                       # "VanillaBERT_ffn_endtoend-200-200-200-200-interaction.model_keywords_r_srand_csT_nnT_it-ig-id_csT_nnT,"
                                       # "VanillaBERT_ffn_endtoend-200-200-200-200-ir_chatgpt_it-ig-id_csT_nnT,"
                                       # "VanillaBERT_ffn_endtoend-200-200-200-200-ir_srand_vocab_it-ig-id_csT_nnT,"
                                       # "VanillaBERT_ffn_endtoend-200-200-200-200-ir_sidf_vocab_it-ig-id_csT_nnT,"
                                       # "VanillaBERT_ffn_endtoend-200-200-200-200-ir_SBERTFULL_vocab_it-ig-id_csT_nnT,"
                                       # "VanillaBERT_ffn_endtoend-200-200-200-200-ir_tf-idf_1_vocab_it-ig-id_csT_nnT,"
                                       # "VanillaBERT_ffn_endtoend-200-200-200-200-ir_tf-idf_3_vocab_it-ig-id_csT_nnT,"
                                      ]
        print_list_col2["weighted"] = ["MF-200-xavier_normal,0.0004", "", "", ""
                                       # "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-ir_srand_csT_nnT_it-ig-id_csT_nnT,"
                                       # "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-ir_sidf_csT_nnT_it-ig-id_csT_nnT,"
                                       # "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-ir_SBERTFULL_csT_nnT_it-ig-id_csT_nnT,"
                                       # "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-ir_tf-idf_1_csFalse_nnT_it-ig-id_csT_nnT,"
                                       # "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-ir_tf-idf_3_csFalse_nnT_it-ig-id_csT_nnT,"
                                       # "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-interaction.model_keywords_r_srand_csT_nnT_it-ig-id_csT_nnT,"
                                       # "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-ir_chatgpt_it-ig-id_csT_nnT,"
                                       # "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-ir_srand_vocab_it-ig-id_csT_nnT,"
                                       # "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-ir_sidf_vocab_it-ig-id_csT_nnT,"
                                       # "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-ir_SBERTFULL_vocab_it-ig-id_csT_nnT,"
                                       # "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-ir_tf-idf_1_vocab_it-ig-id_csT_nnT,"
                                       # "VanillaBERT_ffn_endtoend-embafter-200-200-200-200-200-200-ir_tf-idf_3_vocab_it-ig-id_csT_nnT,"
                                       ]
        print_res()

    print("\end{tabular}")
    print("\end{table*}")

#        for k in print_list:
#            if k in res:
#                if k in name_mapping:
#                    print(f"{name_mapping[k]} & {' & '.join(str(round_half_up(res[k][g]*100, 2)) for g in grps)} \\\\ \hline % {k} ")
#                else:
#                    print(f" & {' & '.join(str(round_half_up(res[k][g]*100, 2)) for g in grps)} \\\\ \hline % {k} ")
#            else:
#                print(f" & \\\\ % {k} ")

#    for k, v in res.items():
#        print(f"{k} & {v} \\")
