import os
import pandas as pd
from eval_cluster_results import do_eval

dir = "./snowball_input/"
results = []
for file in os.listdir(dir):
    if not file.endswith(".csv"):
        continue
    for i, t in [(3, "snowball"),(5, "hdbscan_eom"),(6, "hdbscan_leaf")]:
        args = dict()
        args["clustered"] = dir + file
        args["cluster_method"] = t
        args["clustered_text_column"] = 0
        args["goldstandard"] = dir+file
        args["gold_id_column"] = 2
        args["clustered_id_column"]=i
        args["clustered_report_unclustered_id"] = "-1"
        args["gold_report_unclustered_id"] = "-1"
        args["gold_text_column"] = 0
        p, r, f1, ari, cust = do_eval(args)
        args["precision"] = p
        args["recall"] = r
        args["f1"] = f1
        args["ari"] = ari
        args["cust"] = cust
        results.append(args)
        

df = pd.DataFrame(results)
df.to_csv("./snowball_output/scored_comparison_results.csv")