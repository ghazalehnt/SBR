import argparse
import json
import os
from collections import defaultdict
from os.path import join

from statics import shorten_strategies, shorten_names

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # required: path to gt and pd to be evaluated:
    parser.add_argument('--dir', '-d', type=str, default=None, help='path to exp dirs')
    parser.add_argument('--metric', '-m', type=str, default=None, help='metric')
    parser.add_argument('--group', '-g', type=str, default=None, help='group')
    args, _ = parser.parse_known_args()

    g = args.group
    m = args.metric
    dirs = os.listdir(args.dir)
    res = defaultdict()
    for d in dirs:
        if os.path.exists(join(d, "config.json")) and os.path.exists(join(d, "results_test_neg_standard_100_best_model.txt")):
            config = json.load(open(join(d, "config.json"), 'r'))
            if config['model']['name'].startswith("MF_"):
                n = f"{config['model']['name']}-{config['model']['embedding_dim']}-{config['model']['embed_init']}"
            elif config['model']['name'].startswith("VanillaBERT_ffn_endtoend_"):
                temp1 = config['dataaset']['user_text_file_name']
                for s, v in shorten_strategies.items():
                    temp1 = temp1.replace(s, v)
                for s, v in shorten_names.items():
                    temp1 = temp1.replace(s, v)
                temp2 = config['dataaset']['item_text_file_name']
                for s, v in shorten_strategies.items():
                    temp2 = temp2.replace(s, v)
                for s, v in shorten_names.items():
                    temp2 = temp2.replace(s, v)

                n = f"{config['model']['name']}-{'-'.join(config['model']['user_k'])}-{'-'.join(config['model']['item_k'])}-" \
                    f"{temp1}-{temp2}"
                # if config['model']['embeddingTODO']:
                #     pass
            for line in open(join(d, "results_test_neg_standard_100_best_model.txt"), 'r').readlines():
                if len(line) > 0:
                    r = json.loads(line)
                    if g in r:
                        res[f"{n}-{config['trainer']['lr']}"] = r[g][m]

    for k, v in res.items():
        print(f"{k} & {v} \\")