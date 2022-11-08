import argparse
from collections import defaultdict

import pandas as pd

def main(res_file, given_user_group, given_metric):
    res = pd.read_csv(res_file)
    res = res.fillna("")
    print(res_file)
    print(given_user_group)
    print(given_metric)
    name_mapping = {"BERT_1CH_": "BERT-aggr1",
                    "BERT_5CH_": "BERT-aggr5",
                    "BERT_1CH_idf_sentence": "w-sent",
                    "BERT_1CH_item_sentence_SBERT": "s-sent",
                    "BERT_1CH_tf-idf_3": "w-phrase"}
    lines = defaultdict(lambda: defaultdict())
    for model, user_text, item_text, group, metric in zip(res['model config'], res['user profile'], res['item text'], res['group'], res[given_metric]):
        if given_user_group not in group:
            continue
        if model.startswith("CF"):
            print(f"{model} & {metric} & & & \\\\ \hline % {model}")
        elif model in name_mapping.keys():
            if item_text in ["tc", "tg"]:
                if user_text in ["tc", "tg"]:
                    lines[model][1] = metric
                elif user_text in ["tcsr", "tgr"]:
                    lines[model][3] = metric
            elif item_text in ["tcd", "tgd"]:
                if user_text in ["tc", "tg"]:
                    lines[model][2] = metric
                elif user_text in ["tcsr", "tgr"]:
                    lines[model][4] = metric
    for model in name_mapping.keys():
        print(f"{name_mapping[model]} & {lines[model][1] if 1 in lines[model] else '-'} & {lines[model][2] if 2 in lines[model] else '-'} & {lines[model][3] if 3 in lines[model] else '-'} & {lines[model][4]if 4 in lines[model] else '-'} \\\\ \hline % {model}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_file', '-r', type=str, required=True)
    parser.add_argument('--user_group', '-g', type=str, default="group 1-5")
    parser.add_argument('--report_metric', '-m', type=str, default="ndcg_cut_5")
    args, _ = parser.parse_known_args()
    main(args.res_file, args.user_group, args.report_metric)
