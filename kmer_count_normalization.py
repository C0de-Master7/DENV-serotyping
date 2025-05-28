import pandas as pd
from fileops import file2list


def normalize_kmer_counts(df_raw: pd.DataFrame, kmers, output_file):
    for k in kmers:
        if k not in df_raw.columns:
            df_raw[k] = 0

    def normalize(group):
        total = group[kmers].sum().sum()
        if total == 0:
            return group
        for k in kmers:
            group[k] = group[k] / total
        return group

    df_norm = df_raw.groupby('Sample').apply(normalize).reset_index(drop=True)
    df_norm.to_csv(output_file, index=False)
    print(f"✅ Normalized output written to {output_file}")
    return None


def main():
    kmers = file2list('6m_final_feats.txt')
    input_counts_file = pd.read_csv("raw_kmer_counts_ann.csv")
    output_norm_file = "normalized_kmer_counts_ann.csv"

    normalize_kmer_counts(input_counts_file, kmers, output_norm_file)


if __name__ == "__main__":
    main()
