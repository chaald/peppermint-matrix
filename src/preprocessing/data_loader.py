import pandas as pd

def load_data(file_path:str) -> pd.DataFrame:
    user_array = []
    item_array = []
    with open(file_path) as file_handle:
        for line in file_handle.readlines():
            if len(line) > 0:
                line = line.strip('\n').split(' ')
                items = [int(i) for i in line[1:]]
                uid = int(line[0])

                for item_id in items:
                    user_array.append(uid)
                    item_array.append(item_id)

    user_iteraction = pd.DataFrame({
        "user_id": user_array,
        "item_id": item_array
    })

    return user_iteraction


