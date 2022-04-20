import os
import pandas as pd
from eval_cluster_results import do_eval
from concurrent.futures import ThreadPoolExecutor

n_threads = 32

results = []
def run_task(args):
    p, r, f1, ari, cust = do_eval(args)
    args["precision"] = p
    args["recall"] = r
    args["f1"] = f1
    args["ari"] = ari
    args["cust"] = cust
    results.append(args)

dir = "./snowball_input/"
queries = []

for k, file in enumerate(os.listdir(dir)):
    if not file.endswith(".csv"):
        continue
    # print(k)
    for i, t in [(3, "snowball"),(5, "hdbscan_eom"),(6, "hdbscan_leaf")]:
        for top_k in [1, 4]:
            if "blobs=4" in file and top_k == 4:
                continue
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

with ThreadPoolExecutor(max_workers=n_threads) as pool:
    pool.map(run_task, queries)
# results.append(args)


df = pd.DataFrame(results)
df.to_csv("./snowball_output/scored_top_k_results.csv")