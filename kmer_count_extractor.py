import re
import pandas as pd
import time
import multiprocessing as mp
from functools import partial
from fileops import file2list


def parallel_kmer_count_extraction(params: tuple, kmer_list: list):
    seq = params[1]
    temp_dict = {}
    for kmer in kmer_list:
        pattern = re.compile(kmer)
        result = pattern.findall(seq)
        if len(result) != 0:
            temp_dict[kmer] = len(result)

    temp_dict['y'] = params[2]
    temp_dict['filename'] = params[0]

    return temp_dict


def parallel_kmer_extraction(params: tuple, kmer_list: list, cluster_data: pd.DataFrame):
    seq = params[1]
    temp_dict = {}
    for kmer in kmer_list:
        pattern = re.compile(kmer)
        result = pattern.findall(seq)
        if len(result) != 0:
            # Extracting cluster index of kmer
            c_idx = cluster_data.where(cluster_data == kmer).stack().index.tolist()[0][1]
            if c_idx in list(temp_dict.keys()):
                temp_dict[c_idx].extend([len(result)])
            else:
                temp_dict[c_idx] = [len(result)]

    # Getting count of clusters
    for k, v in temp_dict.items():
        temp = sum(v)
        temp_dict[k] = temp

    temp_dict['y'] = params[2]
    temp_dict['filename'] = params[0]

    return temp_dict


if __name__ == "__main__":
    # Timer
    start_time = time.time()

    # Reading kmer list
    my_kmers = file2list('6mer_features.txt')

    # Reading train data
    train_data = pd.read_parquet('labelled_train_data.parquet')
    train_list = []
    for row in train_data.itertuples(index=True, name="Row"):
        train_list.extend([(row.filename, row.sequence, row.y)])

    # Reading test data
    test_data = pd.read_parquet('labelled_test_data.parquet')
    test_list = []
    for row in test_data.itertuples(index=True, name="Row"):
        test_list.extend([(row.filename, row.sequence, row.y)])

    # Reading cluster file
    # c_data = pd.read_parquet('my_cluster_data_6mer.parquet')

    # Creating partial function
    partial_kmer = partial(parallel_kmer_count_extraction, kmer_list=my_kmers)

    # Creating pool
    with mp.Pool(64) as pool:
        train_results = pool.map(partial_kmer, train_list)
        test_results = pool.map(partial_kmer, test_list)

    # Converting to dataframe
    print("Converting to dataframe")
    train_dict = {}
    for idx in range(len(train_results)):
        train_dict[idx] = train_results[idx]
    train_proc_data = pd.DataFrame.from_dict(train_dict, orient='index')
    train_proc_data.fillna(0, inplace=True)
    train_proc_data.columns = train_proc_data.columns.map(str)

    # Writing dataframe to parquet
    print("Writing dataframe to parquet")
    train_proc_data.to_parquet('train_data_6mer.parquet')

    # Converting to dataframe
    print("Converting to dataframe")
    test_dict = {}
    for idx in range(len(test_results)):
        test_dict[idx] = test_results[idx]
    test_proc_data = pd.DataFrame.from_dict(test_dict, orient='index')
    test_proc_data.fillna(0, inplace=True)
    test_proc_data.columns = test_proc_data.columns.map(str)

    # Writing dataframe to parquet
    print("Writing dataframe to parquet")
    test_proc_data.to_parquet('test_data_6mer.parquet')

    print("Done")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f'Time taken for execution: {execution_time} seconds.')
