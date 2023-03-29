from os.path import join

import pandas as pd
import numpy as np

if __name__ == "__main__":
    datapath = ""
    num_users = 100
    pos_file = "validation.csv"
    new_pos_file = f"validation_u{num_users}.csv"
    neg_file = "validation_neg_random_100.csv"
    new_neg_file = f"validation_neg_u{num_users}_random_100.csv"

    data = pd.read_csv(join(datapath, pos_file), dtype=str)
    users = list(set(data["user_id"]))
    chosen_users = np.random.choice(users, num_users, replace=False)

    new_data = data[data["user_id"].isin(chosen_users)]
    new_data.to_csv(join(datapath, new_pos_file), index=False)

    data = pd.read_csv(join(datapath, neg_file), dtype=str)
    new_data = data[data["user_id"].isin(chosen_users)]
    new_data.to_csv(join(datapath, new_neg_file), index=False)