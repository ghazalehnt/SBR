import argparse
import csv
from os import listdir
from os.path import join


def main(exp_dir, result_file_name):
    rows = []
    for fname in sorted(listdir(exp_dir)):
        print(fname)
        res_file = open(join(exp_dir, fname, result_file_name), 'r')
        reader = csv.reader(res_file)
        header = next(reader)
        for line in reader:
            rows.append([fname] + line)
        rows.append([])

    header = ["config"] + header
    with open(join(args.dir, "test_results.csv")) as outfile:
        print(join(args.dir, "test_results.csv"))
        writer = csv.writer(outfile)
        writer.writerow(header)
        writer.writerows(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', type=str, default=None, help='experiments dir')
    parser.add_argument('--thresholds', type=int, nargs='+', default=None, help='user thresholds')
    parser.add_argument('--set', '-s', type=str, default='test', help='valid/test')
    args, _ = parser.parse_known_args()

    if args.set == 'test':
        res_file_name = f"results_test_th_{'_'.join([str(t) for t in args.thresholds])}.csv"
    elif args.set == 'valid':
        res_file_name = f"results_valid_th_{'_'.join([str(t) for t in args.thresholds])}.csv"
    else:
        raise ValueError("set given wrong!")

    main(args.dir, res_file_name)
