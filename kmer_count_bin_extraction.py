import pandas as pd
from collections import Counter
from fileops import file2list


def range_generator(start, end, interval):
    out = []
    nos = int((end - start) / interval)
    for i in range(nos):
        out.append(((start + (i * interval)), (start + ((i+1) * interval))))
    return out


def count_kmers_in_sequence(sequence: str, kmers: list) -> Counter:
    """Count occurrences of each k-mer in a sequence"""
    k = len(kmers[0])  # assumes all same length
    kmer_counts = Counter()
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if kmer in kmers:
            kmer_counts[kmer] += 1
    return kmer_counts


def count_kmers_in_bin(seq: str, start: int, end: int, kmers: list) -> dict:
    """Extract subsequence from bin and count k-mers"""
    sub_seq = seq[start:end+1]  # inclusive
    counts = count_kmers_in_sequence(sub_seq, kmers)
    return {k: counts.get(k, 0) for k in kmers}


def process_samples_with_bins(parquet_path: str, kmers: list, bins: list) -> pd.DataFrame:
    """Processes k-mer counts per sample and bin"""
    df = pd.read_parquet(parquet_path)

    data = []

    for _, row in df.iterrows():
        sample = row['filename']
        sequence = row['sequence']
        cls = row['y']

        for start, end in bins:
            if end >= len(sequence):
                continue  # skip bin if out of range
            bin_id = f"{start}-{end}"
            kmer_counts = count_kmers_in_bin(sequence, start, end, kmers)
            row_data = {'Sample': sample, 'Class': cls, 'Bin': bin_id}
            row_data.update(kmer_counts)
            data.append(row_data)

    return pd.DataFrame(data)


def main():
    kmers = file2list('6m_final_feats.txt')
    bins = [(97, 435), (436, 927), (928, 2418), (2419, 3474), (3475, 4128), (4129, 4518), (4519, 6375), (6376, 6825),
            (6826, 7572), (7573, 10269)]
    parquet_file = 'labelled_train_data.parquet'
    output_file = "raw_kmer_counts_ann.csv"

    df = process_samples_with_bins(parquet_file, kmers, bins)
    df.to_csv(output_file, index=False)
    print(f"✅ Output saved to: {output_file}")

    print('Done')


if __name__ == "__main__":
    main()
