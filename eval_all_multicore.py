import os
import sys
import pandas as pd
from eval_cluster_results import do_eval
# import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from json_helper import JsonHelper
import psutil
import math
import json

num_cpus = psutil.cpu_count() - 1
# num_cpus = 2

# results = []
def run_task(args, results):
    p, r, f1, ari, cust = do_eval(args)
    args["precision"] = p
    args["recall"] = r
    args["f1"] = f1
    args["ari"] = ari
    args["cust"] = cust
    results.append(args)

def create_split():
    dir = "./snowball_input/"
    queries = []

    for k, file in enumerate(os.listdir(dir)):
        if not file.endswith(".csv"):
            continue
        # print(k)
        for i, t in [(3, "snowball"),(5, "hdbscan_eom"),(6, "hdbscan_leaf")]:
            args = dict()
            args["clustered"] = dir + file
            args["cluster_method"] = t
            args["clustered_text_column"] = 1
            args["goldstandard"] = dir+file
            args["gold_id_column"] = 2
            args["clustered_id_column"]=i
            args["clustered_report_unclustered_id"] = "-1"
            args["gold_report_unclustered_id"] = "-1"
            args["gold_text_column"] = 1
            queries.append(args)

    x = 0
    num_per_cpu = int(math.ceil(len(queries) / num_cpus))
    for i in range(num_cpus):
        batch = queries[x: min(len(queries), x + num_per_cpu)]
        JsonHelper.write_json(batch, f"./snowball_input_batches/batch_{i}.json")


def run_batch(q_batch, i):
    results = []

    for args in q_batch:
        p, r, f1, ari, cust = do_eval(args)
        args["precision"] = p
        args["recall"] = r
        args["f1"] = f1
        args["ari"] = ari
        args["cust"] = cust

    df = pd.DataFrame(results)
    df.to_csv(f"./snowball_output/scored_results_batch_{i}.csv")

if __name__ == "__main__":
    # arg_0 = sys.argv[1]
    # if (arg_0)
    create_split()
    num_batches = len(os.listdir("./snowball_input_batches/"))
    # results = []
    st = ""
    for i in range(num_batches):
        st += " python3 eval_batch.py " + str(i)
        if i != num_batches-1:
            st += " |"
    os.system(st)
    # batch = JsonHelper.parse_json("./snowball_input_batches/batch_{}.json".format(sys.argv[1]))
    # run_batch(batch, sys.argv[1])
    # os.system("python3 eval_all_multicore.py")
# with ThreadPoolExecutor(max_workers=n_threads) as pool:
#     pool.map(run_task, queries)
# results.append(args)


# df = pd.DataFrame(results)
# df.to_csv("./snowball_output/scored_top_k_results.csv")

