import argparse
import json
import math
from os import listdir
from os.path import join, exists, getsize


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


def process_file(fname, eval_set, all_only):
    if not exists(fname):
        print(f"res-paht: {fname}")
        return
    fsize = getsize(fname)
    print(f"res-paht: {fname} - {fsize}")
    if fsize < 3990:
        return
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

    if eval_set == 'valid':
        if all_only:
            res = get_res(valid_res["ALL"], valid_res["WARM"], valid_res["COLD"], all_only=True)
        else:
            res = get_res(valid_res["ALL"], valid_res["WARM"], valid_res["COLD"], all_only=False)
    elif eval_set == 'test':
        if all_only:
            res = get_res(test_res["ALL"], test_res["WARM"], test_res["COLD"], all_only=True)
        else:
            res = get_res(test_res["ALL"], test_res["WARM"], test_res["COLD"], all_only=False)
    print(f"Model: {config['model']}")
    if config['model']['name'] != "MF":
        print(f"Dataset: u{config['dataset']['max_num_chunks_user']}-{config['dataset']['user_text_filter']}-"
              f"{'t' if 'item.title' in config['dataset']['user_text'] else ''},"
              f"{'g' if 'item.genres' in config['dataset']['user_text'] else ''},"
              f"{'r' if 'interaction.review' in config['dataset']['user_text'] else ''}\n"
              f"{config['dataset']['limit_training_data'] if 'limit_training_data' in config['dataset'] else ''}")

    print(res)


def main(exp_dir, res_file_name, eval_set):
    print("----------------ALL-----------------")
    for fname in sorted(listdir(exp_dir)):
        process_file(join(exp_dir, fname, res_file_name), eval_set, True)
        print("-----")
    print("----------------WARM/COLD-------------")
    for fname in sorted(listdir(exp_dir)):
        process_file(join(exp_dir, fname, res_file_name), eval_set, False)
        print("-----")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', type=str, default=None, help='experiments dir')
    parser.add_argument('--res_file_name', '-r', type=str, default='results_coldth5_withtextFalse.txt', help='result file name indicating threshold ...')
    parser.add_argument('--set', '-s', type=str, default='test', help='valid/test')
    args, _ = parser.parse_known_args()
    main(args.dir, args.res_file_name, args.set)












