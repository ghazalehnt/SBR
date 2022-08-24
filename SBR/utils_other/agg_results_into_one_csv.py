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
def main(exp_dir, result_file_name):
    group_rows = {}
    for fname in sorted(listdir(exp_dir)):
        if fname.endswith("csv"):
            continue
        if not exists(join(exp_dir, fname, result_file_name)):
            print(f"no results found for: {fname}")
            continue
        print(fname)
        if fname.startswith("False") or fname.startswith("True"):
            if "test_neg_random_100_1_1_" in fname:
                ch = 1
            elif "test_neg_random_100_5_5_" in fname:
                ch = 5
            else:
                raise ValueError("num chunks not found in fname")
            sortby = ""
            for filter in ["tf-idf_1", "tf-idf_2", "tf-idf_3", "tf-idf_1-2-3", "idf_1_unique", "idf_2_unique",
                           "idf_3_unique", "idf_1-2-3_unique", "idf_sentence"]:
                if filter in fname:
                    sortby = filter
                    break
            profile = []
            if "item.title" in fname:
                profile.append('t')
            if "item.genres" in fname:
                profile.append('g')
            if "interaction.review" in fname:
                profile.append('r')
            profile = ''.join(profile)
            if fname.startswith("True"):
                model_config = f"CF-BERT_{ch}CH_{sortby}"
            else:
                model_config = f"BERT_{ch}CH_{sortby}"
        else:
            model_config = f"CF-{fname[:fname.index('_')]}"
            profile = ''
        res_file = open(join(exp_dir, fname, result_file_name), 'r')
        reader = csv.reader(res_file)
        header = next(reader)
        for line in reader:
            if line[0] not in group_rows:
                group_rows[line[0]] = []
            group_rows[line[0]].append([model_config, profile, line[0]] + [str(round_half_up(float(m)*100, 4)) for m in line[1:]])

    header = ["model config", "user profile"] + header
    with open(join(args.dir, "test_results.csv"), 'w') as outfile:
        print(join(args.dir, "test_results.csv"))
        writer = csv.writer(outfile)
        writer.writerow(header)
        for rows in group_rows.items():
            writer.writerows(rows)
            writer.writerow([])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', type=str, default=None, help='experiments dir')
    parser.add_argument('--thresholds', '-t', type=int, nargs='+', default=None, help='user thresholds')
    parser.add_argument('--set', '-s', type=str, default='test', help='valid/test')
    args, _ = parser.parse_known_args()

    if args.set == 'test':
        res_file_name = f"results_test_th_{'_'.join([str(t) for t in args.thresholds])}.csv"
    elif args.set == 'valid':
        res_file_name = f"results_valid_th_{'_'.join([str(t) for t in args.thresholds])}.csv"
    else:
        raise ValueError("set given wrong!")

    main(args.dir, res_file_name)
