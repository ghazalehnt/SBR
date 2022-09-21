import csv
from collections import defaultdict
from os.path import join

item_replacement = defaultdict()

INTERACTION_FILE = "goodreads_crawled.interactions"
ITEM_FILE = "goodreads_crawled.items"

def main(in_folder, out_folder):
    user_interactions = defaultdict(list)
    item_count = defaultdict(lambda:0)
    item_info = defaultdict()

    with open(join(in_folder, INTERACTION_FILE), 'r') as f:
        reader = csv.reader(f)
        header_inter = next(reader)
        USER_IDX_INTER = header_inter.index("user_id")
        ITEM_IDX_INTER = header_inter.index("item_id")
        RATING_IDX_INTER = header_inter.index("rating")
        REVIEW_IDX_INTER = header_inter.index("review")
        for line in reader:
            user_id = line[USER_IDX_INTER]
            item_id = line[ITEM_IDX_INTER]
            item_count[item_id] += 1
            user_interactions[user_id].append(line)

    with open(join(in_folder, ITEM_FILE), 'r') as f:
        reader = csv.reader(f)
        header_item = next(reader)
        ITEM_IDX_ITEM = header_item.index("item_id")
        AUTHOR_IDX_ITEM = header_item.index("author")
        AVG_RATING_IDX_ITEM = header_item.index("avg_rating")
        for line in reader:
            item_id = line[ITEM_IDX_ITEM]
            item_info[item_id] = line

    print(sum([len(inters) for inters in user_interactions.values()]))
    for user_id in user_interactions.keys():
        user_interactions[user_id] = clean_with_same_review(user_interactions[user_id], item_count, item_info,
                                                            RATING_IDX_INTER, REVIEW_IDX_INTER, ITEM_IDX_INTER,
                                                            AUTHOR_IDX_ITEM, AVG_RATING_IDX_ITEM)
    print(sum([len(inters) for inters in user_interactions.values()]))
    for user_id in user_interactions.keys():
        user_interactions[user_id] = replace_items_by_replacement(user_interactions[user_id], item_info,
                                                                  REVIEW_IDX_INTER, ITEM_IDX_INTER,
                                                                  AUTHOR_IDX_ITEM, AVG_RATING_IDX_ITEM)
    print(sum([len(inters) for inters in user_interactions.values()]))

    all_items = set()
    rows = []
    for user_inter in user_interactions.values():
        for line in user_inter:
            all_items.add(line[ITEM_IDX_INTER])
            rows.append(line)
    rows = sorted(rows, key=lambda x:int(x[0]))
    with open(join(out_folder, INTERACTION_FILE), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header_inter)
        writer.writerows(rows)

    with open(join(out_folder, ITEM_FILE), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header_item)
        for item_id, info_line in item_info.items():
            if item_id in all_items:
                writer.writerow(info_line)


def replace_items_by_replacement(user_inters, item_info,
                                 REVIEW_IDX_INTER, ITEM_IDX_INTER, AUTHOR_IDX_ITEM, AVG_RATING_IDX_ITEM):
    ret_inters = []

    grp_by_final_item = defaultdict(list)
    for line in user_inters:
        item_id = line[ITEM_IDX_INTER]
        if item_id in item_replacement:
            grp_by_final_item[item_replacement[item_id]].append(line)
        else:
            grp_by_final_item[item_id].append(line)

    for item_id, lines in grp_by_final_item.items():
        if len(lines) > 1:
            sorted_lines = sorted(lines, key=lambda x:len(x[REVIEW_IDX_INTER]), reverse=True)
            main_line = sorted_lines[0]
            if main_line[ITEM_IDX_INTER] in item_replacement:
                main_line[ITEM_IDX_INTER] = item_replacement[main_line[ITEM_IDX_INTER]]
            ret_inters.append(main_line)
            for line in sorted_lines[1:]:
                if (item_info[main_line[ITEM_IDX_INTER]][AUTHOR_IDX_ITEM] != item_info[line[ITEM_IDX_INTER]][AUTHOR_IDX_ITEM]) or \
                        (item_info[main_line[ITEM_IDX_INTER]][AVG_RATING_IDX_ITEM] != item_info[line[ITEM_IDX_INTER]][AVG_RATING_IDX_ITEM]):
                    ret_inters.append(line)
        else:
            if lines[0][ITEM_IDX_INTER] in item_replacement:
                lines[0][ITEM_IDX_INTER] = item_replacement[lines[0][ITEM_IDX_INTER]]
            ret_inters.append(lines[0])
    return ret_inters


def clean_with_same_review(user_inters, item_count, item_info,
                           RATING_IDX_INTER, REVIEW_IDX_INTER, ITEM_IDX_INTER, AUTHOR_IDX_ITEM, AVG_RATING_IDX_ITEM):
    ret_inters = []

    # first group them by rating?
    rating_grouped = defaultdict(list)
    for line in user_inters:
        if line[RATING_IDX_INTER] != "" and line[REVIEW_IDX_INTER] != "":
            rating_grouped[line[RATING_IDX_INTER]].append(line)
        else:
            ret_inters.append(line)

    for rating in rating_grouped:
        same_rev = defaultdict(list)
        for line in rating_grouped[rating]:
            same_rev[line[REVIEW_IDX_INTER]].append(line[ITEM_IDX_INTER])

        user_kept_items = []
        for rev, items in same_rev.items():
            if len(items) > 1:
                sorted_items = sorted(items, key=lambda x: item_count[x], reverse=True)
                most_used_item = sorted_items[0]
                user_kept_items.append(most_used_item)
                # if len(rev) > 200:
                #     for item in sorted_items[1:]:
                #         if item in item_replacement and item_replacement[item] != most_used_item:
                #             if item_count[most_used_item] > item_count[item_replacement[item]]:
                #                 item_replacement[item_replacement[item]] = most_used_item
                #                 item_replacement[item] = most_used_item
                #             else:
                #                 item_replacement[most_used_item] = item_replacement[item]
                #         else:
                #             item_replacement[item] = most_used_item
                #     # continue
                # else:
                for item in sorted_items[1:]:
                    if (item_info[most_used_item][AUTHOR_IDX_ITEM] != item_info[item][AUTHOR_IDX_ITEM]) or \
                            (item_info[most_used_item][AVG_RATING_IDX_ITEM] != item_info[item][AVG_RATING_IDX_ITEM]):
                        user_kept_items.append(item)
                    else:
                        if item in item_replacement and item_replacement[item] != most_used_item:
                            if item_count[most_used_item] > item_count[item_replacement[item]]:
                                item_replacement[item_replacement[item]] = most_used_item
                                item_replacement[item] = most_used_item
                            else:
                                item_replacement[most_used_item] = item_replacement[item]
                        else:
                            item_replacement[item] = most_used_item
            else:
                user_kept_items.append(items[0])

        for line in rating_grouped[rating]:
            if line[ITEM_IDX_INTER] in user_kept_items:
                ret_inters.append(line)

    return ret_inters


if __name__ == '__main__':
    input_folder = ""
    output_folder = ""
    main(input_folder, output_folder)