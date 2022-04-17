import pandas as pd
import json

class JsonHelper:

    @staticmethod
    def parse_json(path_with_json):
        with open(path_with_json) as filename:
            try:
                data = json.load(filename)
            except json.decoder.JSONDecodeError:
                print("ERROR: failed to parse " + path_with_json)
                return None
        return data

    @staticmethod
    def write_json(dict_for_json, file_name):
        with open(file_name, 'w') as filename:
            json.dump(dict_for_json, filename, ensure_ascii=False, indent=4)


df = pd.read_json("./snowball_output/snowball_results_19200.json", orient='split', lines=False)
df = df.reset_index()

"""
{
        "num_to_shelf": 4,
        "sim_thresh": 0.93,
        "merge_sim_thresh": 0.87,
        "batch_size": 32,
        "eons": 64,
        "i": 1,
        "size": 10000,
        "std": 1.0,
        "num": 4,
        "is_shuffle": true,
        "classifications": null,
        "class_centers": null,
        "error": 0.005007175160645971
    },
"""
# attributes_dict = dict()
from collections import defaultdict

class HashableKey:
    def __init__(self, d):
        self.num_to_shelf = d["num_to_shelf"]
        self.batch_size = d["batch_size"]
        self.eons = d["eons"]
        self.size = d["size"]
        self.std = d["std"]
        self.is_shuffle = d["is_shuffle"]
        self.d = d

    def __repr__(self):
        return f"{self.num_to_shelf},{self.batch_size},{self.eons},{self.size},{self.std},{self.is_shuffle}"

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


x = defaultdict(list)
for index, row in df.iterrows():
    x[HashableKey(row)].append(row)

def get_std(list_values):
    mean = sum(list_values) / len(list_values)
    variance = sum([((x - mean) ** 2) for x in list_values]) / len(list_values)
    stddev = variance ** 0.5

    print("{0:0.1f}".format(stddev))
    return stddev

new_values = []
for i, value in x.items():
    avg = sum(float(v["error"]) for v in value) / len(value)
    print(i)
    print(avg)
    print()
    z = dict(i.d)
    z["error_mean"] = avg
    z["error_std"] = get_std([float(v["error"]) for v in value])
    new_values.append(z)

# pd.DataFrame(new_values)

JsonHelper.write_json(new_values, "./snowball_output/snowball_results_192.json")
# df.to_csv("./snowball_output/snowball_results_19200.csv")