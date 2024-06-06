import os


def read_data(data_path: str, country: str):
    list_of_client = [
        "_".join(f.split("_")[:-1])
        for f in os.listdir(data_path)
        if (country in f) and ("train" in f)
    ]
    client_dict = {}
    for i, f in enumerate(list_of_client):
        client_dict[i] = {"cid": i, "path": os.path.join(data_path, f)}
    return client_dict
