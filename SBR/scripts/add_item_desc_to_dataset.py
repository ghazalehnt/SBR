### as we crawled the item descriptions later, we want to add them to the dataset.item file in a seperate attempt here.

# load the new file with all descriptions
import csv
from os.path import join

item_desc_file = "TODO/goodreads_crawled_new.items"
item_meta_info = {}
with open(item_desc_file, 'r') as f:
    reader = csv.reader(f)
    header_1 = next(reader)
    ITEM_ID_IDX_1 = header_1.index('item_id')
    ITEM_URL_IDX_1 = header_1.index('item_url')
    TITLE_IDX_1 = header_1.index('title')
    DESC_IDX_1 = header_1.index('description')
    GENRES_IDX_1 = header_1.index('genres')
    AUTHOR_IDX_1 = header_1.index('author')
    AR_IDX_1 = header_1.index('avg_rating')
    NR_IDX_1 = header_1.index('num_rating')
    for line in reader:
        item_meta_info[line[ITEM_ID_IDX_1]] = line

# load the dataset.item
dataset_path = "TODO"
dataset_item_file = join(dataset_path, "items.csv")
items = []
with open(dataset_item_file, 'r') as f:
    reader = csv.reader(f)
    header_2 = next(reader)
    ITEM_ID_IDX_2 = header_2.index('item_id')
    ITEM_URL_IDX_2 = header_2.index('item_url')
    TITLE_IDX_2 = header_2.index('title')
    DESC_IDX_2 = header_2.index('description')
    GENRES_IDX_2 = header_2.index('genres')
    AUTHOR_IDX_2 = header_2.index('author')
    AR_IDX_2 = header_2.index('avg_rating')
    NR_IDX_2 = header_2.index('num_rating')
    for line in reader:
        items.append(line)

# replace the meta info of the item
for item in items:
    item_id = item[ITEM_ID_IDX_2]
    item[TITLE_IDX_2] = item_meta_info[item_id][TITLE_IDX_1]
    item[DESC_IDX_2] = item_meta_info[item_id][DESC_IDX_1]
    item[GENRES_IDX_2] = item_meta_info[item_id][GENRES_IDX_1]
    item[AUTHOR_IDX_2] = item_meta_info[item_id][AUTHOR_IDX_1]
    item[AR_IDX_2] = item_meta_info[item_id][AR_IDX_1]
    item[NR_IDX_2] = item_meta_info[item_id][NR_IDX_1]

# write the item file again
new_dataset_item_file = join(dataset_path, "new_items.csv")
with open(new_dataset_item_file, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(header_2)
    writer.writerows(items)
