import pandas as pd
from fileops import file2list


def aggregate_by_class_bin(input_file: str, kmers: list, output_file: str):
    df = pd.read_csv(input_file)

    required_columns = {'Sample', 'Class', 'Bin'}
    if not required_columns.issubset(df.columns):
        raise ValueError("Input file must contain 'Sample', 'Class', and 'Bin' columns.")

    for k in kmers:
        if k not in df.columns:
            df[k] = 0

    # Group by Class and Bin and compute mean
    agg_df = df.groupby(['Class', 'Bin'])[kmers].mean().reset_index()
    agg_df = agg_df.round(3)

    agg_df.to_csv(output_file, index=False)
    print(f"✅ Aggregated output written to {output_file}")


def main():
    # === CONFIGURATION ===
    kmers = file2list('6m_final_feats.txt')
    input_norm_file = "normalized_kmer_counts_ann.csv"
    output_agg_file = "classwise_kmer_averages_ann.csv"

    aggregate_by_class_bin(input_norm_file, kmers, output_agg_file)


if __name__ == "__main__":
    main()
