import argparse
import json
from collections import defaultdict
from os.path import join

import torch


def main(result_folder):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	item_prec_reps = torch.load(join(result_folder, "item_prec_output"), map_location=device)
	user_prec_reps = torch.load(join(result_folder, "user_prec_output"), map_location=device)
	item_id_mapping = json.load(open(join(result_folder, "item_internal_ids.json"), 'r'))
	user_id_mapping = json.load(open(join(result_folder, "user_internal_ids.json"), 'r'))

	item_id_mapping_rev = {v: k for k, v in item_id_mapping.items()}

	results = defaultdict(lambda: defaultdict())
	for user_exid, user_inid in user_id_mapping.items():
		user_rep = user_prec_reps(torch.LongTensor([user_inid]).to(device))
		res = torch.sigmoid(torch.sum(torch.mul(user_rep, item_prec_reps.weight), dim=1)).tolist()
		for item_inid in range(len(res)):
			results[user_exid][item_id_mapping_rev[item_inid]] = res[item_inid]
		if len(results.keys()) % 1000 == 0:
			print(f"{len(results.keys())} users done.")
	json.dump({"predicted": results}, open(join(result_folder, "all_items_scores_from_precalc.json"), 'w'))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--result_folder', '-r', type=str, default=None, help='result forler, to evaluate')
	args, _ = parser.parse_known_args()
	main(args.result_folder)
