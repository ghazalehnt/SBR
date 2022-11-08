import argparse
import csv
import json
import math
from os import listdir
from os.path import join, exists


def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier + 0.5) / multiplier

metrics = ['ndcg_cut_5', 'ndcg_cut_10', 'ndcg_cut_20']

# False_200_200_dot_product_0.0004_1e-08_256_4096_random_4_f:validation_neg_random_100_f:test_neg_random_100_1_1_512
# _max_pool_mean_last_tf-idf_2_False_True__item.category__item.title-item.category-item-description
def main(exp_dir, evalset):
    group_rows = {}
    for folder_name in sorted(listdir(exp_dir)):
        if folder_name.endswith("csv"):
            continue
        trainer = ""
        user_text = ""
        item_text = ""
        config = json.load(open(join(exp_dir, folder_name, "config.json"), 'r'))
        lr = config['trainer']['lr'] if 'lr' in config['trainer'] else ''
        bs = config['dataset']['train_batch_size'] if 'train_batch_size' in config['dataset'] else ''
        if use_LR is not None and lr != use_LR:
            continue
        if use_BS is not None and bs != use_BS:
            continue

        for result_file_name in listdir(join(exp_dir, folder_name)):
            if not result_file_name.startswith(f"results_{evalset}"):
                continue

            test = result_file_name[:-4]

            if not exists(join(exp_dir, folder_name, result_file_name)):
                print(f"no results found for: {folder_name}")
                continue
            res_file = open(join(exp_dir, folder_name, result_file_name), 'r')
            try:
                reader = csv.reader(res_file)
                header = next(reader)
            except Exception:
                print(f"empty file: {join(exp_dir, folder_name, result_file_name)}")

            model_name = config["model"]["name"]
            if model_name.startswith("MF"):
                model_config = f"{model_name}_{config['model']['embedding_dim']}_{'sig' if config['trainer']['sigmoid_output'] is True else 'no-sig'}"
                trainer = f"{config['trainer']['optimizer']}_{config['trainer']['loss_fn']}"
                training_samples = config['dataset']['training_neg_sampling_strategy']
                validation = config['dataset']['validation_neg_sampling_strategy']
            elif model_name.startswith("VanillaBERT"):
                item_text = ""
                if "item.title" in config['dataset']['item_text']:
                    item_text += "t"
                if "item.category" in config['dataset']['item_text']:
                    item_text += "c"
                if "item.genres" in config['dataset']['item_text']:
                    item_text += "g"
                if "item.description" in config['dataset']['item_text']:
                    item_text += "d"

                user_text = ""
                if "item.title" in config['dataset']['user_text']:
                    user_text += "t"
                if "item.category" in config['dataset']['user_text']:
                    user_text += "c"
                if "item.genres" in config['dataset']['user_text']:
                    user_text += "g"
                if "interaction.summary" in config['dataset']['user_text']:
                    user_text += "s"
                if "interaction.review_text" in config['dataset']['user_text']:
                    user_text += "r"
                if "interaction.reviewText" in config['dataset']['user_text']:
                    user_text += "r"

                ch = config['dataset']['max_num_chunks_user']
                sortby = config['dataset']['user_text_filter']
                if model_name == "VanillaBERT_precalc_embed_sim":
                    model_config = f"emb-sim_"
                    trainer = ""
                    training_samples = ""
                    validation = ""
                else:
                    trainer = f"{config['trainer']['optimizer']}_{config['trainer']['loss_fn']}"
                    training_samples = config['dataset']['training_neg_sampling_strategy']
                    validation = config['dataset']['validation_neg_sampling_strategy']
                    if model_name == "VanillaBERT_precalc_with_ffn":
                        model_config = f"ffn{config['model']['k1']}-{config['model']['k2']}-dp_"
                    elif model_name == "VanillaBERT_precalc_with_itembias":
                        model_config = f"itembias-dp_"
                    elif model_name == "VanillaBERT_precalc_with_ffn_itembias":
                        model_config = f"ffn{config['model']['k1']}-{config['model']['k2']}-itembias-dp_"
                if config['model']['use_CF'] is True:
                    model_config += "CF_"
                model_config += f"BERT_{ch}CH_{sortby}_{'sig' if config['trainer']['sigmoid_output'] is True else 'no-sig'}"

            for line in reader:
                if line[0] not in group_rows:
                    group_rows[line[0]] = []
                file_line = [model_config, user_text, item_text, line[0]]
                for metric in metrics:
                    m = 0
                    if metric in header:
                        IDX = header.index(metric)
                        if line[IDX] != '':
                            m = str(round_half_up(float(line[IDX])*100, 2))
                    file_line.append(m)
                file_line.extend([lr, bs, trainer, training_samples, validation, test, folder_name])
                group_rows[line[0]].append(file_line)

    header = ["model config", "user profile", "item text", "user group"] + metrics + ["lr", "bs", "trainer", "training", "validation", "test", "path"]
    outf = f"{evalset}_results" \
           f"{f'_bs{use_BS}' if use_BS is not None else ''}" \
           f"{f'_lr{use_LR}' if use_LR is not None else ''}.csv"
    with open(join(args.dir, outf), 'w') as outfile:
        print(join(args.dir, outf))
        writer = csv.writer(outfile)
        writer.writerow(header)
        for rows in group_rows.values():
            writer.writerows(rows)
            writer.writerow([])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', type=str, default=None, help='experiments dir')
    # parser.add_argument('--file_suffix', '-f', type=str, default=None, help='suffix of eval file')
    parser.add_argument('--set', '-s', type=str, default='test', help='valid/test')
    parser.add_argument('--bs', '-b', type=str, default=None, help='bs')
    parser.add_argument('--lr', '-l', type=str, default=None, help='lr')
    args, _ = parser.parse_known_args()
    use_BS = args.bs
    use_LR = args.lr

    if args.set not in ["test", "valid"]:
        raise ValueError("set given wrong!")

    main(args.dir, args.set)
