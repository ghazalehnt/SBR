import json
import math
from os import listdir
from os.path import join


def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier + 0.5) / multiplier


def get_line(result, metrics):
    line = []
    for m in metrics:
        line.append(str(round_half_up(float(result[m])*100, 2)))
    return " & ".join(line)


def get_res(all, warm, cold, all_only=True):
    if all_only:
        ordered_metrics = ['P_1', 'Rprec', 'ndcg_cut_5', 'ndcg_cut_10', 'ndcg_cut_20', 'f1_micro', 'f1_macro']
        res = all
        return get_line(res, ordered_metrics)

    ordered_metrics = ['P_1', 'Rprec', 'ndcg_cut_5', 'ndcg_cut_10', 'ndcg_cut_20']
    # warm
    res = warm
    l1 = get_line(res, ordered_metrics)
    # cold
    res = cold
    l2 = get_line(res, ordered_metrics)
    return l1 + "  &  " + l2


def process_file(fname):
    valid_res = {"ALL": {}, "WARM": {}, "COLD": {}}
    test_res = {"ALL": {}, "WARM": {}, "COLD": {}}
    config = json.load(open(join(fname[:fname.rindex("/")+1], "config.json"), 'r'))
    with open(fname, 'r') as f:
        for line in f:
            if line.startswith("Valid results ALL"):
                valid_res["ALL"] = json.loads(line[line.index("{"):].replace("'", '"'))
            elif line.startswith("Valid results WARM"):
                valid_res["WARM"] = json.loads(line[line.index("{"):].replace("'", '"'))
            elif line.startswith("Valid results COLD"):
                valid_res["COLD"] = json.loads(line[line.index("{"):].replace("'", '"'))
            elif line.startswith("Test results ALL"):
                test_res["ALL"] = json.loads(line[line.index("{"):].replace("'", '"'))
            elif line.startswith("Test results WARM"):
                test_res["WARM"] = json.loads(line[line.index("{"):].replace("'", '"'))
            elif line.startswith("Test results COLD"):
                test_res["COLD"] = json.loads(line[line.index("{"):].replace("'", '"'))

    res_all = get_res(valid_res["ALL"], valid_res["WARM"], valid_res["COLD"], all_only=True)
    res_wc = get_res(valid_res["ALL"], valid_res["WARM"], valid_res["COLD"], all_only=False)
    print(f"Model: {config['model']}")
    if config['model']['name'] != "MF":
        print(f"Dataset: u{config['dataset']['max_num_chunks_user']}-{config['dataset']['user_text_filter']}")
    print(f"res-paht: {fname}")
    print(res_all)
    print(res_wc)


exp_dir = ''
for fname in listdir(exp_dir):
    process_file(join(exp_dir, fname))
    print("-----")

















