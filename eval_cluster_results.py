import argparse
import csv
from collections import defaultdict


def binom(n, k):
	if k < 0 or k > n:
		return 0
	if k > (n / 2):
		k = k - n
	
	denom = 1.0
	num = 1.0
	for i in range(1, k+1):
		denom *= i
		num *= (n + 1 - i)
	
	return num / denom


def eval():

	parser = argparse.ArgumentParser(description="""Evaluate clustering results. This scripts will return the
	                                                precision, recall, F1-score and ARI of the provided clustering result""")
	
	parser.add_argument("--clustered",
	                    required=True,
	                    type=str,
	                    help="Path to the .csv file with the clustering result. 12th column is text, 10th column is cluster id.")

	parser.add_argument("--clustered_text_column",
						required=False,
						type=int,
						default=12,
						help="12th column of cluster report is text by default.")

	parser.add_argument("--clustered_id_column",
	                    required=False,
	                    type=int,
	                    default=10,
	                    help="10th column of cluster report is cluster id by default.")

	parser.add_argument("--clustered_report_unclustered_id",
						required=False,
						type=str,
						default="-1",
						help="-1 string is unclustered id by default")
	
	parser.add_argument("--goldstandard",
	                    required=False,
	                    type=str,
	                    help="path to the .csv file with the gold standard. 1st column is text, 2nd column is cluster id/label.")
	
	parser.add_argument("--gold_text_column",
	                    required=False,
	                    type=int,
	                    default=0,
	                    help="0th column of gold report is text by default.")
	
	parser.add_argument("--gold_id_column",
	                    required=False,
	                    type=int,
	                    default=1,
	                    help="1st column of gold report is cluster id by default.")
	
	parser.add_argument("--gold_report_unclustered_id",
	                    required=False,
	                    type=str,
	                    default="-1",
	                    help="-1 string is gold unclustered id by default")
	
	
	args = vars(parser.parse_args())
	
	do_eval(args)

def do_eval(args):
	cluster_text_col = args["clustered_text_column"]
	cluster_id_col = args["clustered_id_column"]
	cluster_unclustered_id = args["clustered_report_unclustered_id"]

	gold_text_col = args["gold_text_column"]
	gold_id_col = args["gold_id_column"]
	gold_unclustered_id = args["gold_report_unclustered_id"]

	if "top_k" in args:
		top_k = args["top_k"]
	else:
		top_k = None

	# utterance_map = JsonHelper.parse_json(args["processmap"])

	# Get paths to clustering result and gold standard
	clustered_file = args["clustered"]
	gold_standard_file = args["goldstandard"]
	if gold_standard_file is None:
		gold_standard_file = clustered_file

	print("\nStarting evaluate.py...")
	print("Clustering results file:", clustered_file)
	print("gold standard file:", gold_standard_file)

	# Get the number of clusters from the gold standard
	# ---------------------------------------------------------

	all_gold_standard_clusters_list = []

	with open(gold_standard_file) as csvfile:
		readCSV = csv.reader(csvfile, delimiter=',')
		for i, row in enumerate(readCSV):
			if i == 0:
				continue
			if row[gold_id_col].strip() == gold_unclustered_id:
				continue
			all_gold_standard_clusters_list.append(row[gold_id_col].strip())

	gold_standard_clusters_list = list(set(all_gold_standard_clusters_list))
	gold_standard_n_clusters = len(gold_standard_clusters_list)

	print("\nNumber of clusters available in the gold standard: ", gold_standard_n_clusters)

	# Get the gold standard
	# ----------------------------
	gold_standard_clusters = [[] for x in range(gold_standard_n_clusters)]

	gold_standard_count = 0

	unclustered_gold = 0

	with open(gold_standard_file) as contig_clusters:
		readCSV = csv.reader(contig_clusters, delimiter=',')

		for i, row in enumerate(readCSV):
			if i == 0:
				continue
			if row[gold_id_col].strip() == gold_unclustered_id:
				unclustered_gold += 1
				continue
			gold_standard_count += 1
			contig = row[gold_text_col].strip().lower()
			bin_num = gold_standard_clusters_list.index(row[gold_id_col].strip())
			gold_standard_clusters[bin_num].append(contig)

	print("Number of objects available in the gold standard: ", gold_standard_count)

	# Get the number of clusters from the initial clustering result
	# ---------------------------------------------------------

	all_clusters_list = []

	with open(clustered_file) as csvfile:
		readCSV = csv.reader(csvfile, delimiter=',')
		for i, row in enumerate(readCSV):
			if i == 0:
				continue
			if row[cluster_id_col].strip() == cluster_unclustered_id:
				continue
			all_clusters_list.append(row[cluster_id_col].strip())

	clusters_list = list(set(all_clusters_list))
	n_clusters = len(clusters_list)

	print("Number of clusters available in the clustering result: ", n_clusters)

	# Get initial clustering result
	# ----------------------------
	clusters = [[] for x in range(n_clusters)]

	clustered_count = 0
	unclustered_count = 0
	clustered_objects = []

	size_per_cluster = defaultdict(int)

	with open(clustered_file) as contig_clusters:
		readCSV = csv.reader(contig_clusters, delimiter=',')
		for i, row in enumerate(readCSV):
			if i == 0:
				continue
			if row[cluster_id_col].strip() == cluster_unclustered_id:
				unclustered_count += 1
				continue
			clustered_count += 1
			contig = row[cluster_text_col].strip().lower()
			bin_num = clusters_list.index(row[cluster_id_col].strip())
			size_per_cluster[bin_num] += 1
			clusters[bin_num].append(contig)
			clustered_objects.append(contig)

	if top_k:
		print("top K: " + str(top_k))
		sorted_items = sorted(size_per_cluster.items(), key=lambda item: item[1])
		sorted_items.reverse()
		# sorted_items = [(key, value) for key, value in sorted_items.items()]
		new_clusters = dict()
		clustered_objects = list()
		clustered_count = 0
		for i, (key, value) in enumerate(sorted_items):
			if i >= top_k:
				break
			clustered_count += value
			new_clusters[i] = clusters[key]
			clustered_objects.extend(new_clusters[i])
		clusters = new_clusters
		n_clusters = len(clusters)

	print("Number of objects available in the clustering result: ", len(clustered_objects))

	# Functions to determine precision, recall, F1-score and ARI
	# ------------------------------------------------------------

	# Get precicion
	def getPrecision(mat, k, s, total):
		sum_k = 0
		for i in range(k):
			max_s = 0
			for j in range(s):
				if mat[i][j] > max_s:
					max_s = mat[i][j]
			sum_k += max_s
		return sum_k / total * 100

	# Get recall
	def getRecall(mat, k, s, total, unclassified):
		sum_s = 0
		for i in range(s):
			max_k = 0
			for j in range(k):
				if mat[j][i] > max_k:
					max_k = mat[j][i]
			sum_s += max_k

		return sum_s / (total + unclassified) * 100

	# Get ARI
	# clusters_species, n_clusters, gold_standard_n_clusters, total_clustered
	def getARI(mat, k, s, N):
		t1 = 0
		for i in range(k):
			sum_k = 0
			for j in range(s):
				sum_k += mat[i][j]
			t1 += binom(sum_k, 2)

		t2 = 0
		for i in range(s):
			sum_s = 0
			for j in range(k):
				sum_s += mat[j][i]
			t2 += binom(sum_s, 2)

		t3 = t1 * t2 / binom(N, 2)

		t = 0
		for i in range(k):
			for j in range(s):
				t += binom(mat[i][j], 2)

		ari = (t - t3) / ((t1 + t2) / 2 - t3) * 100
		return ari

	# Get F1-score
	def getF1(prec, recall):
		return 2 * prec * recall / (prec + recall)

	# Determine precision, recall, F1-score and ARI for clustering result
	# ------------------------------------------------------------------

	total_clustered = 0

	clusters_species = [[0 for x in range(gold_standard_n_clusters)] for y in range(n_clusters)]

	for i in range(n_clusters):
		for j in range(gold_standard_n_clusters):
			n = 0
			for k in range(clustered_count):
				if clustered_objects[k] in clusters[i] and clustered_objects[k] in gold_standard_clusters[j]:
					n += 1
					total_clustered += 1
			clusters_species[i][j] = n

	print("Number of objects available in the clustering result that are present in the gold standard:",
		  total_clustered)

	my_precision = getPrecision(clusters_species, n_clusters, gold_standard_n_clusters, total_clustered)
	my_recall = getRecall(clusters_species, n_clusters, gold_standard_n_clusters, total_clustered,
						  (gold_standard_count - total_clustered))
	try:
		my_ari = getARI(clusters_species, n_clusters, gold_standard_n_clusters, total_clustered)
	except:
		my_ari = 0

	try:
		my_f1 = getF1(my_precision, my_recall)
	except:
		my_f1 = 0

	print("\nEvaluation Results:")
	print("Precision =", my_precision)
	print("Recall =", my_recall)
	print("F1-score =", my_f1)
	print("ARI =", my_ari)

	custom_score = 0
	if (my_ari + my_f1 > 0):
		custom_score = (2 * my_ari * my_f1) / (my_ari + my_f1)

	print("Custom Score =", custom_score)

	print()
	return (my_precision, my_recall, my_f1, my_ari, custom_score)

if __name__ == "__main__":
	# # For debugging
	# import sys
	# commd = [""] + "--goldstandard /users/davidwiner/Downloads/ManualAnnotation.csv --clustered /users/davidwiner/Documents/workspace/cluster_job.csv".split()
	# sys.argv = commd
	
	eval()