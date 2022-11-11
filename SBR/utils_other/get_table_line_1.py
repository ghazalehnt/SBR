import argparse
from collections import defaultdict

import pandas as pd


def main(res_file, given_user_group, given_metric, given_test):
    res = pd.read_csv(res_file)
    res = res.fillna("")
    print(res_file)
    print(given_user_group)
    print(given_metric)
    lines = defaultdict(lambda: defaultdict())
    for model, user_text, item_text, group, metric, test in \
            zip(res['model config'], res['user profile'], res['item text'],
                res['user group'], res[given_metric], res['test']):
        if given_user_group not in group:
            continue
        if given_test != test:
            continue
        if model.startswith("MF"):
            print(f"{model} & {metric} & & & \\\\ \hline % {model}")
        else:
            if item_text in ["tc", "tg"]:
                if user_text in ["tc", "tg"]:
                    print(f"basic: {res['lr']}-{res['bs']}")
                    lines[model][1] = metric
                elif user_text in ["tcsr", "tgr"]:
                    print(f"+rev: {res['lr']}-{res['bs']}")
                    lines[model][3] = metric
                elif user_text in ["sr", "r"]:
                    print(f"tg/tc-r/sr {res['lr']}-{res['bs']}")
                    lines[model][5] = metric
            elif item_text in ["tcd", "tgd"]:
                if user_text in ["tc", "tg"]:
                    print(f"+desc: {res['lr']}-{res['bs']}")
                    lines[model][2] = metric
                elif user_text in ["tcsr", "tgr"]:
                    print(f"full: {res['lr']}-{res['bs']}")
                    lines[model][4] = metric
                elif user_text in ["sr", "r"]:
                    print(f"tgd/tcd-r/sr {res['lr']}-{res['bs']}")
                    lines[model][6] = metric
        print("\n")
        print(f"{model} & "
              f"{lines[model][1] if 1 in lines[model] else '-'} & "
              f"{lines[model][2] if 2 in lines[model] else '-'} & "
              f"{lines[model][3] if 3 in lines[model] else '-'} & "
              f"{lines[model][4]if 4 in lines[model] else '-'}  & "
              f"{lines[model][5] if 3 in lines[model] else '-'} & "
              f"{lines[model][6] if 3 in lines[model] else '-'} "
              f"\\\\ \hline % {model}")
        print("\n\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_file', '-r', type=str, required=True)
    parser.add_argument('--user_group', '-g', type=str, default="group 1-5")
    parser.add_argument('--report_metric', '-m', type=str, default="ndcg_cut_5")
    parser.add_argument('--report_test', '-t', type=str, default="results_test_th_5_50_random_100")
    args, _ = parser.parse_known_args()
    main(args.res_file, args.user_group, args.report_metric, args.report_test)
