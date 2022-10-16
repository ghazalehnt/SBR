import argparse
import csv
import math
from os import listdir
from os.path import join, exists


def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier + 0.5) / multiplier

# False_200_200_dot_product_0.0004_1e-08_256_4096_random_4_f:validation_neg_random_100_f:test_neg_random_100_1_1_512
# _max_pool_mean_last_tf-idf_2_False_True__item.category__item.title-item.category-item-description
def main(exp_dir, evalset, file_suffix):
    if evalset not in ["test", "valid"]:
        raise ValueError("set given wrong!")

    result_file_name = f"results_{evalset}_{file_suffix}.csv"

    group_rows = {}
    for folder_name in sorted(listdir(exp_dir)):
        if folder_name.endswith("csv"):
            continue

        lr = ""
        if "4e-05_1e-08" in folder_name:
            lr = "0.00004"
        elif "0.0004_1e-08" in folder_name:
            lr = "0.0004"
        elif "0.004_1e-08" in folder_name:
            lr = "0.004"
        elif "0.04_1e-08" in folder_name:
            lr = "0.04"
        elif "0.4_1e-08" in folder_name:
            lr = "0.4"
        elif "0.01_1e-08" in folder_name:
            lr = "0.01"
        elif "0.001_1e-08" in folder_name:
            lr = "0.001"
        elif "0.1_1e-08" in folder_name:
            lr = "0.1"
        if use_LR is not None and lr != use_LR:
            continue

        bs = ""
        if "1e-08_256" in folder_name:
            bs = "256"
        elif "1e-08_128" in folder_name:
            bs = "128"
        elif "1e-08_64" in folder_name:
            bs = "64"
        elif "1e-08_512" in folder_name:
            bs = "512"
        elif "1e-08_32" in folder_name:
            bs = "32"
        if use_BS is not None and bs != use_BS:
            continue

        if not exists(join(exp_dir, folder_name, result_file_name)):
            print(f"no results found for: {folder_name}")
            continue
#        print(folder_name)

        res_file = open(join(exp_dir, folder_name, result_file_name), 'r')
        try:
            reader = csv.reader(res_file)
            header = next(reader)
        except Exception:
            print(f"empty file: {join(exp_dir, folder_name, result_file_name)}")

        valid_neg = ""
        if "f-random_100_f-random_100" in folder_name or "f-random_100_f-genres_100" in folder_name:
            valid_neg = "random"
        elif "f-genres_100_f-genres_100" in folder_name or "f-genres_100_f-random_100" in folder_name:
            valid_neg = "genres"

        if folder_name.startswith("False") or folder_name.startswith("True"):
            if "_100_1_1_" in folder_name:
                ch = 1
            elif "_100_5_5_" in folder_name:
                ch = 5
            else:
                print(folder_name)
                raise ValueError("num chunks not found in fname")
            sortby = ""
            for filter in ["tf-idf_1", "tf-idf_2", "tf-idf_3", "tf-idf_1-2-3", "idf_1_unique", "idf_2_unique",
                               "idf_3_unique", "idf_1-2-3_unique", "idf_sentence", "item_sentence_SBERT",
                           "random_sentence", "item_per_chunk"]:
                if filter in folder_name:
                    sortby = filter
                    break

            item_text = ""
            item_text_part = folder_name[folder_name.rindex("_"):]
            if item_text_part == "_item.title-item.category-item.description":
                item_text = "tcd"
            elif item_text_part == "_item.title-item.genres-item.description":
                item_text = "tgd"
            elif item_text_part == "_item.title-item.category":
                item_text = "tc"
            elif item_text_part == "_item.title-item.genres":
                item_text = "tg"
            if item_text != "":
                folder_name = folder_name[:folder_name.rindex("_")]

            profile = []
            if "item.title" in folder_name:
                profile.append('t')
            if "item.genres" in folder_name:
                profile.append('g')
            if "item.category" in folder_name:
                profile.append('c')
            if "interaction.summary" in folder_name:
                profile.append('s')
            if "interaction.review" in folder_name:
                profile.append('r')
            profile = ''.join(profile)
            if folder_name.startswith("True"):
                model_config = f"CF-BERT_{ch}CH_{sortby}"
            else:
                model_config = f"BERT_{ch}CH_{sortby}"
        elif folder_name.startswith("MF"):
            if folder_name.startswith("MF_with_itembias"):
                folder_name = folder_name[folder_name.index('MF_with_itembias_')+len('MF_with_itembias_'):]
                model_config = f"MF_with_itembias_{folder_name[:folder_name.index('_')]}"
            elif folder_name.startswith("MF"):
                folder_name = folder_name[folder_name.index('MF_') + len('MF_'):]
                model_config = f"MF_{folder_name[:folder_name.index('_')]}"
            profile = ''
            item_text = ''
        else:
            raise ValueError(folder_name)
        book_limit = 'all books'
        if 'max_book' in folder_name:
            temp = folder_name[folder_name.index("max_book_")+len("max_book_"):]
            book_pick_strategy = temp[temp.index("_")+1:]
            num_books = temp[:temp.index("_")]
            book_limit = f"max {num_books} - {book_pick_strategy}"



        for line in reader:
            if line[0] not in group_rows:
                group_rows[line[0]] = []
            group_rows[line[0]].append([model_config, profile, item_text, book_limit, line[0]]
                                       + [str(round_half_up(float(m)*100, 4)) for m in line[1:]]
                                       + [lr, bs])

    header = ["model config", "user profile", "item text", "book limit"] + header + ["lr", "bs"]
    outf = f"{evalset}_results_{file_suffix}" \
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
    parser.add_argument('--file_suffix', '-f', type=str, default=None, help='suffix of eval file')
    parser.add_argument('--set', '-s', type=str, default='test', help='valid/test')
    parser.add_argument('--bs', '-b', type=str, default=None, help='bs')
    parser.add_argument('--lr', '-l', type=str, default=None, help='lr')
    args, _ = parser.parse_known_args()
    use_BS = args.bs
    use_LR = args.lr

    main(args.dir, args.set, args.file_suffix)
