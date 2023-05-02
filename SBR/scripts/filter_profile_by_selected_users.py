import random
from os.path import join

import pandas as pd
import numpy as np

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    new_datapath = ""
    original_profile_root_path = ""
    profiles = []

    users = pd.read_csv(join(new_datapath, "users.csv"))
    filtered_users = list(users["user_ud"])
    for profile_file in profiles:
        data = pd.read_csv(join(original_profile_root_path, profile_file), dtype=str)
        new_data = data[data["user_id"].isin(filtered_users)]
        new_data.to_csv(join(new_datapath, profile_file), index=False)
