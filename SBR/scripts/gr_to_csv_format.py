import csv
import gzip
import json
from os.path import join
import pandas as pd


def main(interaction_input_file, item_meta_input_file, item_genres_input_file, output_path, database_name):
    # load user-item interactions:
    interactions = []
    users = set()
    header_inter = ["user_id", "book_id", "rating", "review_text", "date_added", "date_updated", "read_at", "started_at", "n_votes", "n_comments"]
    with gzip.open(interaction_input_file) as f:
        for line in f:
            r = json.loads(line)
            interactions.append([str(r[h]).replace("\n", " ").strip() if h in r else "" for h in header_inter])
            users.add(r["user_id"])

    with open(join(output_path, f"{database_name}.interactions"), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header_inter)
        writer.writerows(interactions)

    # load item meta:
    data = []
    with gzip.open(item_meta_input_file) as f:
        for line in f:
            data.append(json.loads(line.strip()))
    item_info = pd.DataFrame.from_dict(data)
    item_info = item_info.drop_duplicates(subset=['book_id'])
    item_info = item_info.fillna('')
    item_info = item_info.replace(r'\n', ' ', regex=True)
    item_info["authors"] = item_info["authors"].apply(lambda x: ",".join(sorted([a['author_id'].strip() for a in x])))

    data = []
    with gzip.open(item_genres_input_file) as f:
        for line in f:
            data.append(json.loads(line.strip()))
    item_genres = pd.DataFrame.from_dict(data)
    item_genres = item_genres.drop_duplicates(subset=['book_id'])
    item_genres = item_genres.fillna('')
    item_genres = item_genres.replace(r'\n', ' ', regex=True)
    item_genres['genres'] = item_genres['genres'].apply(lambda x: [k.replace(",", "") for k in x.keys()])

    item_info = item_info.merge(item_genres, on="book_id", how="left")

    print(len(item_info))
    item_info.to_csv(join(output_path, f"{database_name}.items"), index=False)

    with open(join(output_path, f"{database_name}.users"), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["user_id"])
        writer.writerows([[k] for k in users])


if __name__ == '__main__':
    input_folder = ""
    output_folder = ""

    inter_file = "goodreads_reviews_dedup.json.gz"
    item_meta_file = "goodreads_books.json.gz"  # title, author_id->list, average_rating, description
    item_genres_file = "goodreads_book_genres_initial.json.gz"
    name = "goodreads_uscd"

    main(join(input_folder, inter_file), join(input_folder, item_meta_file), join(input_folder, item_genres_file),
         output_folder, name)