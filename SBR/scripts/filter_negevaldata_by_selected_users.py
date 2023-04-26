import random
from os.path import join

import pandas as pd
import numpy as np

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    datapath = ""
    num_users = 1000
    st = "standard"
    filtered_pos_file = f"validation_u{num_users}.csv"
    neg_file = f"validation_neg_{st}_100.csv"
    new_neg_file = f"validation_neg_u{num_users}_{st}_100.csv"

    data = pd.read_csv(join(datapath, filtered_pos_file), dtype=str)
    users = list(set(data["user_id"]))

    data = pd.read_csv(join(datapath, neg_file), dtype=str)
    new_data = data[data["user_id"].isin(users)]
    new_data.to_csv(join(datapath, new_neg_file), index=False)