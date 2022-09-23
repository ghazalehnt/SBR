import csv
import gzip
import json
from collections import defaultdict
from os.path import join
import pandas as pd


def main(interaction_input_file, item_meta_input_file, output_path, database_name):
    # load user-item interactions:
    interactions = []
    user_names = defaultdict()
    header_inter = ["reviewerID", "asin", "reviewerName", "vote", "style", "reviewText", "overall", "summary", "unixReviewTime", "reviewTime", "image"]
    with gzip.open(interaction_input_file) as f:
        for line in f:
            r = json.loads(line)
            interactions.append([r[h].replace("\n", " ").strip() if h in r else "" for h in header_inter])
            user_names[r["reviewerID"]] = r["reviewerName"] if "reviewerName" in r else ""

    with open(join(output_path, f"{database_name}.interactions"), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header_inter)
        writer.writerows(interactions)
    # do more on interactions:
    df_inter = pd.read_csv(join(output_path, f"{database_name}.interactions"))
    df_inter = df_inter.replace(r'\n', ' ', regex=True)
    # sort by time and then drop the duplicate keeping last one
    df_inter = df_inter.sort_values(by=['unixReviewTime'])
    df_inter = df_inter.drop_duplicates(subset=['reviewerID', 'asin'], keep='last')
    df_inter.to_csv(join(output_path, f"{database_name}.interactions"))

    # load item meta:
    # header_item = ["asin", "title", "feature", "description", "price", "imageURL", "imageURLHighRes", "related", "salesRank", "brand", "categories", "tech1", "tech2", "similar"]
    data = []
    with gzip.open(item_meta_input_file) as f:
        for line in f:
            data.append(json.loads(line.strip()))
    item_info = pd.DataFrame.from_dict(data)
    print(len(item_info))
    item_info = item_info.fillna('')
    item_info = item_info[~item_info.title.str.contains('getTime')] # filter those unformatted rows (from their colab notebook)
    item_info = item_info.replace(r'\n', ' ', regex=True)
    # remove duplicate asins
    item_info = item_info.drop_duplicates(subset=['asin'])
    print(len(item_info))
    item_info.to_csv(join(output_path, f"{database_name}.items"), index=False)

    with open(join(output_path, f"{database_name}.users"), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["reviewerID", "reviewerName"])
        writer.writerows([[k, v.replace("\n", " ").strip()] for k, v in user_names.items()])


if __name__ == '__main__':
    input_folder = ""
    output_folder = ""

    inter_file = "Books.json.gz"  # OR Books_5.json.gz ?
    item_meta_file = "meta_Books.json.gz"
    name = "amazon_reviews_books"

    main(join(input_folder, inter_file), join(input_folder, item_meta_file), output_folder, name)