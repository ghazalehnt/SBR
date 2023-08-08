import argparse
import json
from os.path import join

import torch

from SBR.utils.data_loading import load_split_dataset, get_user_used_items

topN = 100

def main(result_folder):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	item_prec_reps = torch.load(join(result_folder, "item_prec_output"), map_location=device)
	user_prec_reps = torch.load(join(result_folder, "user_prec_output"), map_location=device)
	item_id_mapping = json.load(open(join(result_folder, "item_internal_ids.json"), 'r'))
	user_id_mapping = json.load(open(join(result_folder, "user_internal_ids.json"), 'r'))
	item_id_mapping_rev = {v: k for k, v in item_id_mapping.items()}

	# reading train and valid to remove from items:
	config = json.load(open(join(result_folder, "config.json"), 'r'))
	datasets, user_info, item_info, _ = load_split_dataset(config["dataset"], False)
	user_used_items = get_user_used_items(datasets, {})
	not_include_in_test = user_used_items['train'].copy()
	for user_id, u_items in user_used_items['validation'].items():
		not_include_in_test[user_id] = not_include_in_test[user_id].union(u_items)

	results = {}
	for user_exid, user_inid in user_id_mapping.items():
		temp = {}
		user_rep = user_prec_reps(torch.LongTensor([user_inid]).to(device))
		res = torch.sigmoid(torch.sum(torch.mul(user_rep, item_prec_reps.weight), dim=1)).tolist()
		for item_inid in range(len(res)):
			if item_inid in not_include_in_test[user_inid]:
				continue
			temp[item_inid] = res[item_inid]
		temp = sorted(temp.items(), key=lambda x: x[1], reverse=True)
		temp_idx = [x[0] for x in temp]
		last_idx = 0
		for item_inid in user_used_items["test"][user_inid]:
			idx = temp_idx.index(item_inid)
			if idx > last_idx:
				last_idx = idx
		if topN > last_idx:
			last_idx = topN
		results[user_exid] = {item_id_mapping_rev[k]: v for k, v in temp[:last_idx]}
		if len(results.keys()) % 1000 == 0:
			print(f"{len(results.keys())} users done.")

	json.dump(results, open(join(result_folder, f"scores_sorted_of_all_items_per_user.json"), 'w'))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--result_folder', '-r', type=str, default=None, help='result forler, to evaluate')
	args, _ = parser.parse_known_args()
	main(args.result_folder)