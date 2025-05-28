import time
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from fileops import list2file, file2list


def main():
    # Timer
    start_time = time.time()

    X = pd.read_parquet('train_data_6mer_features.parquet')
    # print(X[['filename', 'y']])
    kmer_list = X.drop(columns=['filename', 'y']).columns.to_list()
    y = X['y']
    X.drop(columns=['filename'], inplace=True)

    k = 500  # Number of features to select

    # -------- Step 1: Normalize data for ANOVA --------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -------- Step 2: Apply ANOVA F-test --------
    anova_selector = SelectKBest(score_func=f_classif, k=k)
    X_anova_selected = anova_selector.fit_transform(X_scaled, y)
    anova_features = anova_selector.get_support(indices=True)

    # -------- Step 3: Apply Mutual Information --------
    mi_selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_mi_selected = mi_selector.fit_transform(X, y)
    mi_features = mi_selector.get_support(indices=True)

    # -------- Step 4: Compare overlap --------
    anova_set = set(anova_features)
    mi_set = set(mi_features)
    overlap = anova_set.intersection(mi_set)

    print(f"ANOVA selected: {len(anova_set)} features")
    print(f"MI selected: {len(mi_set)} features")
    print(f"Overlap: {len(overlap)} features")
    print(f"Percent overlap: {len(overlap) / k * 100:.2f}%")

    # -------- Optional: View top overlapping features --------
    overlap_list = sorted(list(overlap))
    overlapping_kmers = [kmer_list[i] for i in overlap_list]
    list2file(overlapping_kmers, 'stat_kmer_list_v1.txt')
    print(f"Indices of overlapping features: {overlap_list[:10]} ...")

    km_list = file2list('stat_kmer_list.txt')
    ov_km_list = list(set(km_list) & set(overlapping_kmers))
    print(f"Overlap: {len(ov_km_list)} features")
    print(f"Percent overlap: {len(ov_km_list) / k * 100:.2f}%")
    list2file(overlapping_kmers, 'stat_kmer_list_cm.txt')

    end_time = time.time()
    execution_time = end_time - start_time
    print(f'Time taken for execution: {execution_time} seconds.')


if __name__ == "__main__":
    main()
