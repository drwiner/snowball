import os
import pandas as pd
from eval_cluster_results import do_eval

dir = "./snowball_input/"
results = []
for k, file in enumerate(os.listdir(dir)):
    if not file.endswith(".csv"):
        continue
    print(k)
    for i, t in [(3, "snowball"),(5, "hdbscan_eom"),(6, "hdbscan_leaf")]:
        for top_k in [1, 4]:
            args = dict()
            if "blobs=4" in file and top_k == 4:
                continue
            args["clustered"] = dir + file
            args["cluster_method"] = t
            args["clustered_text_column"] = 1
            args["goldstandard"] = dir+file
            args["gold_id_column"] = 2
            args["clustered_id_column"]=i
            args["clustered_report_unclustered_id"] = "-1"
            args["gold_report_unclustered_id"] = "-1"
            args["gold_text_column"] = 1
            args["top_k"] = top_k
            p, r, f1, ari, cust = do_eval(args)
            args["precision"] = p
            args["recall"] = r
            args["f1"] = f1
            args["ari"] = ari
            args["cust"] = cust
            results.append(args)
        

df = pd.DataFrame(results)
df.to_csv("./snowball_output/scored_top_k_results.csv")