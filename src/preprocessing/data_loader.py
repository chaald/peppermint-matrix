import pandas as pd

from typing import Tuple, Dict

def load_data(file_path:str) -> pd.DataFrame:
    user_array = []
    item_array = []
    user_items = {}
    with open(file_path) as file_handle:
        for line in file_handle.readlines():
            if len(line) > 0:
                line = line.strip('\n').split(' ')
                items = [int(i) for i in line[1:]]
                uid = int(line[0])

                user_items[uid] = items

                for item_id in items:
                    user_array.append(uid)
                    item_array.append(item_id)

    user_iteraction = pd.DataFrame({
        "user_id": user_array,
        "item_id": item_array
    })

    return user_iteraction


