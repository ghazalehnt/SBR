import argparse
import csv
import math
from os import listdir
from os.path import join, exists


def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier + 0.5) / multiplier

# False_200_dot_product_0.0004_1e-08_256_4096_random_4_f:validation_neg_random_100_f:test_neg_random_100_1_1_512_max_pool_mean_last__False_True_item.avg_rating_interaction.review
# False_200_dot_product_0.0004_1e-08_256_4096_random_4_f:validation_neg_random_100_f:test_neg_random_100_1_1_512_max_pool_mean_last__False_True_item.avg_rating_item.title-item.genres,Test
# .avg_rating_item.title-item.genres-interaction.review,Test
def main(exp_dir, evalset, file_suffix):
    if evalset == 'test':
        result_file_name = f"results_test_{file_suffix}.csv"
    elif evalset == 'valid':
        result_file_name = f"results_valid_{file_suffix}.csv"
    else:
        raise ValueError("set given wrong!")

    group_rows = {}
    for folder_name in sorted(listdir(exp_dir)):
        if folder_name.endswith("csv"):
            continue
        if not exists(join(exp_dir, folder_name, result_file_name)):
            print(f"no results found for: {folder_name}")
            continue
#        print(folder_name)
        if folder_name.startswith("False") or folder_name.startswith("True"):
            if "test_neg_random_100_1_1_" in folder_name:
                ch = 1
            elif "test_neg_random_100_5_5_" in folder_name:
                ch = 5
            else:
                raise ValueError("num chunks not found in fname")
            sortby = ""
            for filter in ["tf-idf_1", "tf-idf_2", "tf-idf_3", "tf-idf_1-2-3", "idf_1_unique", "idf_2_unique",
                           "idf_3_unique", "idf_1-2-3_unique", "idf_sentence", "item_sentence_SBERT"]:
                if filter in folder_name:
                    sortby = filter
                    break
            profile = []
            if "item.title" in folder_name:
                profile.append('t')
            if "item.genres" in folder_name:
                profile.append('g')
            if "interaction.review" in folder_name:
                profile.append('r')
            profile = ''.join(profile)
            if folder_name.startswith("True"):
                model_config = f"CF-BERT_{ch}CH_{sortby}"
            else:
                model_config = f"BERT_{ch}CH_{sortby}"
        else:
            model_config = f"CF-{folder_name[:folder_name.index('_')]}"
            profile = ''
        book_limit = 'all books'
        if 'max_book' in folder_name:
            temp = folder_name[folder_name.index("max_book_")+len("max_book_"):]
            book_pick_strategy = temp[temp.index("_")+1:]
            num_books = temp[:temp.index("_")]
            book_limit = f"max {num_books} - {book_pick_strategy}"

        res_file = open(join(exp_dir, folder_name, result_file_name), 'r')
        try:
            reader = csv.reader(res_file)
            header = next(reader)
        except Exception:
            print(f"empty file: {res_file}")
        for line in reader:
            if line[0] not in group_rows:
                group_rows[line[0]] = []
            group_rows[line[0]].append([model_config, profile, book_limit, line[0]] + [str(round_half_up(float(m)*100, 4)) for m in line[1:]])

    header = ["model config", "user profile", "book limit"] + header
    with open(join(args.dir, f"{evalset}_results_{file_suffix}.csv"), 'w') as outfile:
        print(join(args.dir, f"{evalset}_results_{file_suffix}.csv"))
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
    args, _ = parser.parse_known_args()

    main(args.dir, args.set, args.file_suffix)
