import pandas as pd


df = pd.read_json("./snowball_output/snowball_results_192.json")
df.to_csv("./snowball_output/snowball_results_192.csv")