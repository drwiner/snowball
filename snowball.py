import math
import argparse
import metadict
import random
import json
import numpy as np
import pandas as pd
import hdbscan
from sklearn.datasets import make_blobs

from collections import deque, defaultdict
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})


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


class Cluster:
    def __init__(self, item):
        self.count = 1
        self.sum = list(item)
        self.center = list(item)
        self.age = 0

    def assign(self, item):
        self.count += 1
        for i in range(len(item)):
            self.sum[i] += item[i]
            self.center[i] = self.sum[i] / self.count

    def merge(self, other_cluster):
        self.count += other_cluster.count
        for i in range(len(other_cluster.sum)):
            self.sum[i] += other_cluster.sum[i]
            self.center[i] = self.sum[i] / self.count

    def __str__(self):
        return f"Cluster(count={self.count}, age={self.age})"

    def __repr__(self):
        return str(self)

class Repo:
    def __init__(self, clusters=None):
        if not clusters:
            self.clusters = []
            self.size = 0
        else:
            self.clusters = sorted(clusters, key=lambda x: x.count, reverse=True)
            self.size = len(self.clusters)

    def tail(self):
        self.sort()
        return self.clusters[-1]

    def sort(self):
        self.clusters.sort(key=lambda x: x.count, reverse=True)

    def pop(self):
        if self.size == 0:
            return None
        last = self.tail()
        self.clusters = self.clusters[:-1]
        self.size = self.size - 1
        return last

    def push(self, cluster):
        self.clusters.append(cluster)
        self.size += 1

    def batch(self, batch_size):
        if self.size == 0:
            return None
        for i in range(0, self.size, batch_size):
            yield self.clusters[i: min(i+batch_size, len(self.clusters))]

    def __str__(self):
        return f"Repo(size={self.size})"

    def __repr__(self):
        return str(self)


def sim(a, b):
    # numerator = a[0] * b[0] + a[0] * b[1] + a[1] * b[0] + a[1] * b[1]
    # denom = math.sqrt(sum(i**2 for i in a)) * math.sqrt(sum(i **2 for i in b))
    # result = math.acos(numerator / denom)
    # return result
    # return cosine_similarity(a, b)
    return 1 - (math.hypot(b[0] - a[0], b[1] - a[1]))




def snowball(original_data, args, reprocess=None, existing_shelves=None, existing_purgatory=None):

    data = [list(item) for item in original_data]
    """

    :param data:
    :param args:
    :return:
    """

    if existing_purgatory:
        purgatory = existing_purgatory
        start_int = 0
    else:
        # Initially, get first batch size
        purgatory = Repo([Cluster(datum) for datum in data[:args.batch_size]])
        start_int = args.batch_size+1

    if existing_shelves:
        shelf = existing_shelves
    else:
        shelf = Repo()

    reprocess_queue = deque(maxlen=args.eons * args.batch_size)

    # Starting from the subsequent batch, looking at every batch size input:
    for i, index in enumerate(range(start_int, len(data), args.batch_size)):
        # next batch of data
        new_data_batch = data[index: min(index + args.batch_size, len(data))]


        # Processing step Part 1
        if shelf.size > 0:
            # Accumulate Shelf with this batch first, leaving remainder
            remainder = []
            for item in new_data_batch:
                best_shelf = -1
                best_sim = -2
                for j, saved_item in enumerate(shelf.clusters):
                    sim_score = sim(item, saved_item.center)
                    if sim_score < args.sim_thresh:
                        continue
                    if sim_score > best_sim:
                        best_shelf = j
                        best_sim = sim_score
                if best_shelf > -1:
                    shelf.clusters[best_shelf].assign(item)
                else:
                    remainder.append(item)
        else:
            remainder = new_data_batch

        # Processing step part 2: Accumulate purgatory leave remainder
        for item in remainder:
            best_purg = -1
            best_sim = -2

            for j, cluster in enumerate(purgatory.clusters):
                sim_score = sim(item, cluster.center)
                if sim_score < args.sim_thresh:
                    continue
                if sim_score > best_sim:
                    best_sim = sim_score
                    best_purg = j

            if best_purg > -1:
                purgatory.clusters[best_purg].assign(item)
            else:
                # Processing stage step 3: transform to singletons
                purgatory.push(Cluster(item))


        # maintenance stage
        boots = []
        still_alive = []
        for purg_cluster in purgatory.clusters:
            purg_cluster.age += 1
            if purg_cluster.age > args.eons:
                if len(shelf.clusters) < args.num_to_shelf:
                    shelf.push(purg_cluster)
                else:
                    boots.append(purg_cluster)
            else:
                still_alive.append(purg_cluster)

        purgatory = Repo(still_alive)

        if shelf.size == 0 or not boots:
            continue

        merge_or_replace(shelf, boots, purgatory, args, reprocess_queue, reprocess=reprocess)

    return (shelf, purgatory, reprocess_queue)


def merge_or_replace(shelf, boots, purgatory, args, reprocess_queue, reprocess=None):
    weakest_shelf = shelf.tail()
    old_shelves = []

    # First, give priority to purgatory if they have greater count.
    for boot in boots:
        if boot.count > weakest_shelf.count:
            old_shelf = shelf.pop()
            old_shelves.append(old_shelf)
            shelf.push(boot)
            # This could be the boot again.
            weakest_shelf = shelf.tail()
        else:
            # If the boot is going to be discarded, try to merge with existing shelf first.
            if not decide_merge_clusters(boot, shelf.clusters, args):
                if not decide_merge_clusters(boot, purgatory.clusters, args):
                    if reprocess:
                        reprocess_queue.append(boot)

    # Second, for each item that was deshelved, find closest shelf and merge.
    for old_shelf in old_shelves:
        if not decide_merge_clusters(old_shelf, shelf.clusters, args):
            if not decide_merge_clusters(old_shelf, purgatory.clusters, args):
                if reprocess:
                    reprocess_queue.append(old_shelf)

def decide_merge_clusters(cluster, candidate_receivers, args):
    best_cluster = -1
    best_sim = -2
    for w, receiver in enumerate(candidate_receivers):
        sim_score = sim(cluster.center, receiver.center)
        if sim_score >= args.merge_sim_thresh:
            if sim_score > best_sim:
                best_cluster = w
                best_sim = sim_score

    if best_cluster > -1:
        closest_cluster = candidate_receivers[best_cluster]
        closest_cluster.merge(cluster)
        return True
    return False

def main(data, args, do_plot=True, plot_ax=None):
    shelf, purgatory, reprocess_queue = snowball(data, args, reprocess=True)

    for cluster in reprocess_queue:
        if not decide_merge_clusters(cluster, shelf.clusters, args):
            decide_merge_clusters(cluster, purgatory.clusters, args)

    # shelf.sort()
    # plt.plot([item[0] for item in data], [item[1] for item in data], ".", color="grey")
    # plt.plot([x.center[0] for x in shelf.clusters], [x.center[1] for x in shelf.clusters], ".", color='red')
    #
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # plt.legend(numpoints=1)
    # plt.show()

    # reshape queue into batches
    # reprocess_data = list(reprocess_queue)
    # print(f"Reprocessing: {len(reprocess_data)}")
    # shelf, purgatory, _ = snowball(reprocess_data, args, reprocess=None, existing_shelves=shelf,
    #                                existing_purgatory=purgatory)

    """
    For any remaining purgatory cluster, if it hasn't aged yet but has larger, merge or replace.
    """
    merge_or_replace(shelf, purgatory.clusters, purgatory, args, reprocess_queue, reprocess=None)

    # Now we got our shelves, we need to go through entire dataset again
    cluster_list = [[] for i in range(len(shelf.clusters) + 1)]
    label_list = []
    cluster_list_indices = set()
    centers = []
    for i, item in enumerate(data):
        best_cluster = -1
        best_sim = -2
        for j, cluster in enumerate(shelf.clusters):
            sim_score = sim(cluster.center, item)
            if sim_score >= args.sim_thresh:
                if sim_score > best_sim:
                    best_cluster = j
                    best_sim = sim_score

        if best_cluster > -1:
            cluster_list[best_cluster].append(item)
            label_list.append(best_cluster)
            centers.append(shelf.clusters[best_cluster])
        else:
            cluster_list_indices.add(i)
            label_list.append(-1)
            cluster_list[-1].append(item)
            centers.append(-1)

    error = 0
    for i, center in args.class_centers.items():
        highest_sim = -2
        for cluster in shelf.clusters:
            sim_score = sim(center, cluster.center)
            if sim_score > highest_sim:
                highest_sim = sim_score
        error += (1 - highest_sim)
    print(f"ERROR: {error/args.num_to_shelf}")
    if not do_plot:
        return error, label_list, centers
    if not plot_ax:
        plot_ax = plt
    for i, cluster in enumerate(shelf.clusters):
        data_points = cluster_list[i]
        # print(i)
        # mark = marker[i]
        color = (random.random(), random.random(), random.random())
        plot_ax.plot([x[0] for x in data_points], [x[1] for x in data_points], ".", color=color)

        plot_ax.plot([cluster.center[0]], [cluster.center[1]], "o", label=f"({len(data_points)})", color=color)

    # args.classifications:
    markers = ['<', '>', '^', '*', '.']
    for i in range(args.num_to_shelf):
        sub_indices = np.where(args.classifications==i)
        indices = [j for j in sub_indices[0] if j in cluster_list_indices]
        sub_data = data[indices]
        plot_ax.plot([x[0] for x in sub_data], [x[1] for x in sub_data], markers[i%len(markers)], color='grey')
    # for i, datum in enumerate(data):
    #     if i in cluster_list_indices:
    #         continue

        # plt.plot([x[0] for x in cluster_list[-1]], [x[1] for x in cluster_list[-1]], ".", color='grey')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    # plot_ax.legend(numpoints=1)
    plot_ax.set_title("Snowball")
    plt.show()
    # else:
    #     plt.plot([x[0] for x in cluster_list[-1]], [x[1] for x in cluster_list[-1]], ".", color='grey')
    #     plt.xlim(0, 1)
    #     plt.ylim(0, 1)
    #     plt.legend(numpoints=1)
    #     plt.show()

    # plt.plot([x[0] for x in data], [x[1] for x in data], "^", color='grey')
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # plt.show()

    print("STOP")

#     circle1 = plt.Circle((0, 0), 0.2, color='r')
# plt.gca().add_patch(circle1)

def test_3(do_plots=True):
    # Hdbscan test...
    sizes = [1000, 2000, 10000, 50000]
    stds = [1.0, 1.2, 1.5, 2.0]
    num_blobs = [4, 10, 50, 100, 200]

    error_results = []
    count = 0
    # Number of samples per cluster method
    for size in sizes:
        print(size)
        for std in stds:
            for num in num_blobs:
                for is_shuffle in [True, False]:
                    args = metadict.MetaDict()
                    args.num_to_shelf = num
                    args.sim_thresh = 0.93
                    args.merge_sim_thresh = 0.87
                    args.batch_size = 32
                    args.eons = min(int(math.floor(size / 40)), 64)
                    args.size = size
                    args.std = std
                    args.num = num
                    args.is_shuffle = is_shuffle

                    params = (args.sim_thresh, args.merge_sim_thresh, args.batch_size, args.eons)

                    # hdbscan_eom = []
                    # hdbscan_leaf = []
                    # snowball = []
                    # For each (average error, std error, average error top 4, std error top 4, average errror top 1, std error top 1)
                    for i in range(100):
                        args.i = i
                        blobs = make_blobs(size, centers=num, cluster_std=std, shuffle=is_shuffle, return_centers=True, random_state=i)

                        data = prep_from_blobs(blobs, args)

                        hdbscan_clusterer = hdbscan.HDBSCAN(cluster_selection_epsilon=num / size, cluster_selection_method='leaf', min_cluster_size=int(math.floor(math.sqrt((size/num)))))
                        labels = hdbscan_clusterer.fit_predict(data)



                        eom_clusterer = hdbscan.HDBSCAN(cluster_selection_epsilon=num / size)
                        eom_labels = eom_clusterer.fit_predict(data)
                        error, snowball_labels, snowball_centers = main(data, args, do_plot=do_plots)

                        results = list()
                        for w in range(len(data)):
                            result = dict()
                            result["datum"] = data[w]
                            result["blob"] = args.classifications[w]
                            result["snowball_label"] = snowball_labels[w]
                            cluster = snowball_centers[w]
                            if type(cluster) is Cluster:
                                result["snowball_center"] = cluster.center
                            else:
                                result["snowball_center"] = cluster
                            result["hdbscan_eom_label"] = eom_labels[w]
                            result["hdbscan_leaf_label"] = labels[w]
                            results.append(result)

                        df = pd.DataFrame(results)
                        df.to_csv(f"./snowball/snowball_input/size={size}_std={std}_blobs={num}_shuffle={is_shuffle}_params={params}_i={i}.csv")

                        if do_plots:
                            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

                            cluster_dict = defaultdict(list)

                            for w in range(len(args.classifications)):
                                cluster_dict[labels[w]].append(data[w])

                            for key, val_list in cluster_dict.items():
                                if key == -1:
                                    color = 'grey'
                                else:
                                    color = (random.random(), random.random(), random.random())
                                ax2.plot([x[0] for x in val_list], [x[1] for x in val_list], ".", color=color)

                            ax2.set_title(r"HDBSCAN (leaf, min_size=$\sqrt{\frac{n}{k}}$)")
                            ax1.set_title(r"HDBSCAN (eom, $\epsilon=\frac{k}{n}$)")


                            cluster_dict_eom = defaultdict(list)
                            for w in range(len(args.classifications)):
                                cluster_dict_eom[eom_labels[w]].append(data[w])

                            for key, val_list in cluster_dict_eom.items():
                                if key == -1:
                                    color = 'grey'
                                else:
                                    color = (random.random(), random.random(), random.random())
                                ax1.plot([x[0] for x in val_list], [x[1] for x in val_list], ".", color=color)

                            fig.suptitle("HDBSCAN versus Snowball")

                        # plt.show()
                        args.error = error
                        # error_results.append(args)
                        count += 1
        # print(f"Count: {count}")

    JsonHelper.write_json(error_results, "./snowball_output/comparison_results.json")

def test_2():
    # args = metadict.MetaDict()
    # args.batch_size = 16
    # args.eons = 32
    # args.num_to_shelf = 4
    # args.sim_thresh = 0.93
    # args.merge_sim_thresh = 0.87

    sizes = [1000, 2000, 10000, 50000]
    stds = [1.0, 1.2, 1.5, 2.0]
    num_blobs = [4, 6, 10, 20, 50, 100]

    error_results = []
    count= 0

    for size in sizes:
        for std in stds:
            for num in num_blobs:
                for is_shuffle in [True, False]:
                    # Number of samples per cluster method
                    for i in range(100):
                        args = metadict.MetaDict()
                        args.num_to_shelf = num
                        args.sim_thresh = 0.93
                        args.merge_sim_thresh = 0.87
                        args.batch_size = 32
                        args.eons = min(int(math.floor(size/40)), 64)
                        args.i = i
                        args.size = size
                        args.std = std
                        args.num = num
                        args.is_shuffle=is_shuffle

                        blobs = make_blobs(size, centers=num, cluster_std=std, shuffle=is_shuffle,
                                           return_centers=True, random_state=i)

                        data = prep_from_blobs(blobs, args)
                        error, snowball_labels, _ = main(data, args, do_plot=False)
                        args.error = error
                        args.classifications = None
                        args.class_centers = None
                        error_results.append(args)
                        count += 1
                        # JsonHelper.write_json(error_results, "snowball_output/snowball_results.json")
                        # exit(1)
        print(f"Count: {count}")

    JsonHelper.write_json(error_results, "snowball/snowball_output/snowball_results.json")


def prep_from_blobs(blobs, args):
    xmin = blobs[0].min()
    xmax = blobs[0].max()

    data = blobs[0]
    args.classifications = blobs[1]
    # Put each item into [0, 1] range
    ran = (xmax - xmin)
    sub = xmin / ran
    denom = np.broadcast_to(ran, data.shape)
    sub_term = np.broadcast_to(sub, data.shape)
    data = np.subtract(np.divide(data, denom), sub_term)

    cluster_centers = dict()
    for i, center in enumerate(blobs[2]):
        new_center = np.divide(np.subtract(center, xmin), ran)
        cluster_centers[i] = new_center
    args.class_centers = cluster_centers

    return data

def test_1():
    args = metadict.MetaDict()
    args.batch_size=16
    args.eons =32
    args.num_to_shelf = 4
    args.sim_thresh=0.93
    args.merge_sim_thresh=0.87

    blobs = make_blobs(2000, centers=args.num_to_shelf, cluster_std=1.5, shuffle=True, return_centers=True, random_state=12)
    xmin = blobs[0].min()
    xmax = blobs[0].max()

    data = blobs[0]
    args.classifications = blobs[1]
    # Put each item into [0, 1] range
    ran = (xmax - xmin)
    sub = xmin / ran
    denom = np.broadcast_to(ran, data.shape)
    sub_term = np.broadcast_to(sub, data.shape)
    data = np.subtract(np.divide(data, denom), sub_term)

    cluster_centers = dict()
    for i, center in enumerate(blobs[2]):
        new_center = np.divide(np.subtract(center, xmin), ran)
        cluster_centers[i] = new_center
    args.class_centers = cluster_centers


    # get cluster centers for real
    # cluster_centers = dict()
    # for i in range(args.num_to_shelf):


    # data = []
    # for i in range(1500):
    #     rando_x = random.random()
    #     rando_y = random.random()
    #     data.append([rando_x, rando_y])

    # plt.plot([x[0] for x in data], [x[1] for x in data], "^", color='grey')
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # plt.show()
    error, snowball_labels, _ = main(data, args)

    # clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
    # make_blobs(100)


if __name__ == "__main__":
    test_3(do_plots=False)
    # parser = argparse.ArgumentParser("Snowball Cluster Algorithm")
    # parser.add_argument("--input_data_path", required=True, type=str)
    # parser.add_argument("--input_data_type", default="python", type=str)
    # parser.add_argument("--batch_size", default=16, type=int)
    # parser.add_argument("--eons", default=16, type=int)
    # parser.add_argument("--sim_thresh", default=0.9, type=float)
    # parser.add_argument("--merge_sim_thresh", default=0.85, type=float)
    # parser.add_argument("--num_to_shelf", default=8, type=int)
    # parser.add_argument("--output_path", default="./snowball_output/", type=str)
    #
    # args = parser.parse_args()
    #
    # data = []
    # with open(args.input_data_path, 'r') as file:
    #     for line in file:
    #         if not line:
    #             continue
    #         if args.input_data_type == "python":
    #             result = eval(line)
    #         elif args.input_data_type == "comma":
    #             result = [float(i.strip()) for i in line.split(",")]
    #         else:
    #             result = [float(i.strip()) for i in line.split(" ")]
    #         data.append(result)
    #
    #
    #
    # main(data, args)