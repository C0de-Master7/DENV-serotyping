import pandas as pd
import re
import random
import time
from evolutionary_models import hky85_distance_score, hky85_parameters
from sklearn import cluster
from fileops import file2list
import numpy as np
from evolutionary_models import get_consensus
from Bio.Align import PairwiseAligner


def kmerExtractor(data, k=6, opt='list', kmer_list=None):
    # Options for extracting kmer list or kmer counts
    opt_values = ['list', 'count']
    if opt not in opt_values:
        raise ValueError(f"Invalid argument: {opt}. Allowed values are {opt_values}")
    if opt == 'count' and kmer_list is None:
        raise ValueError(f"Empty argument: kmer_list needs to be given to extract count")

    if opt == 'count':
        seq_dict = {}
        for i in range(len(data)):
            # seq = data.iloc[i]['sequence']
            # file = data.iloc[i]['filename']
            seq = data[i]
            temp_dict = {}
            for kmer in kmer_list:
                pattern = re.compile(kmer)
                result = pattern.findall(seq)
                temp_dict[kmer] = len(result)
            # seq_dict[file] = temp_dict
            seq_dict[i] = temp_dict

        # Converting to dataframe
        data = pd.DataFrame(seq_dict)
        data = data.transpose()
        data.reset_index(level=0, inplace=True)
        data = data.drop(columns=['index'])
        data = data.fillna(0)
        return data

    else:
        kmers = []
        for i in range(len(data)):
            seq = data.iloc[i]['sequence']
            for j in range(len(seq) - k + 1):
                temp = seq[j:j + k]
                if not re.search('[RYSWKMBDHVN]', temp) and temp not in kmers:
                    kmers.append(temp)
        return kmers


def mykmeans(feature_list, k=128, max_iters=100):
    # Step 1: Randomly initialize centroids
    random.seed(7)
    centroids = random.sample(feature_list, k=k)
    cluster_dict = {}

    for iteration in range(max_iters):
        # Step 2: Assign points to the nearest centroid
        if iteration == 0:
            # df = pd.read_parquet('distance_6mers.parquet')
            rem_features = list(set(feature_list) - set(centroids))
            for feature in rem_features:
                distance_scores = []
                for centroid in centroids:
                    p, q, pi_a, pi_c, pi_g, pi_t = hky85_parameters(feature, centroid)
                    distance_scores.append(hky85_distance_score(p, q, pi_a, pi_c, pi_g, pi_t))
                min_ds_index = np.argmin(distance_scores)
                c = centroids[min_ds_index]
                new_centroid = get_consensus(feature, c)
                centroids[min_ds_index] = new_centroid
                if min_ds_index not in list(cluster_dict.keys()):
                    cluster_dict[min_ds_index] = [feature, c]
                else:
                    cluster_dict[min_ds_index].extend([feature])

        else:
            new_centroids = centroids.copy()
            new_cluster_dict = {}
            for feature in feature_list:
                distance_scores = []
                for centroid in new_centroids:
                    p, q, pi_a, pi_c, pi_g, pi_t = hky85_parameters(feature, centroid)
                    distance_scores.append(hky85_distance_score(p, q, pi_a, pi_c, pi_g, pi_t))
                min_ds_index = np.argmin(distance_scores)
                c = new_centroids[min_ds_index]
                new_centroid = get_consensus(feature, c)
                new_centroids[min_ds_index] = new_centroid
                if min_ds_index in list(new_cluster_dict.keys()):
                    new_cluster_dict[min_ds_index].extend([feature])
                else:
                    new_cluster_dict[min_ds_index] = [feature]

            # Check for convergence
            aligner = PairwiseAligner()
            i = 0
            for c, nc in zip(centroids, new_centroids):
                if aligner.score(c, nc) >= 4:
                    i += 1
            if i == 128:
                print(f"Converged in {iteration + 1} iterations!")
                break

            centroids = new_centroids

    return centroids, cluster_dict


def normal_clustering(data: list, n_clusters=128):
    # Encoding data
    encoded = []
    vocab = {'A': 1, 'T': 2, 'G': 3, 'C': 4}
    for kmer in data:
        temp = [vocab.get(base, vocab) for base in kmer]
        encoded.append(temp)

    # Clustering until convergence
    kmeans = cluster.KMeans(n_clusters, random_state=7)
    cluster_data = list(kmeans.fit_predict(encoded))
    return cluster_data


def main():
    # Timer
    start_time = time.time()

    # Reading feature list
    kmers = file2list('features_6mer.txt')

    # custom kmer clustering
    centroids, cluster_dict = mykmeans(kmers)
    my_cluster_data = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in cluster_dict.items()]))
    my_cluster_data.to_parquet('my_cluster_data_6mer.parquet')

    # conventional kmeans clustering
    centroids, cluster_dict = normal_clustering(kmers)
    conventional_cluster_data = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in cluster_dict.items()]))
    conventional_cluster_data.to_parquet('conventional_cluster_data_6mer.parquet')

    end_time = time.time()
    execution_time = end_time - start_time
    print(f'Time taken for execution: {execution_time} seconds.')


if __name__ == "__main__":
    main()
