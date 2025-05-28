from sklearn import model_selection, metrics, ensemble
import pandas as pd
import numpy as np
from fileops import file2list
import time


def cross_val_loop(model, data, splits=10):
    # Reading input data
    y = data['y']
    x = data.drop(columns=['y'])

    # Initialising cross validation data splitter
    skf = model_selection.StratifiedKFold(n_splits=splits, shuffle=True, random_state=7)
    scores = {}
    accuracy, balanced_accuracy, precision, recall, f1, roc_auc = [], [], [], [], [], []
    re_x_train = None

    # Cross validation loop
    for _, (train_index, test_index) in enumerate(skf.split(x, y)):
        # Creating train test split data
        x_train, y_train = x.iloc[train_index], y.iloc[train_index]
        re_x_train = x_train
        x_test, y_test = x.iloc[test_index], y.iloc[test_index]

        # Model training and prediction
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)

        # Model evaluation
        accuracy.append(metrics.accuracy_score(y_test, y_pred))
        balanced_accuracy.append(metrics.balanced_accuracy_score(y_test, y_pred))
        precision.append(metrics.precision_score(y_test, y_pred, average='weighted'))
        recall.append(metrics.recall_score(y_test, y_pred, average='weighted'))
        f1.append(metrics.f1_score(y_test, y_pred, average='weighted'))
        roc_auc.append(metrics.roc_auc_score(y_test, y_prob, average='weighted', multi_class='ovo'))

    # Appending scores
    scores['accuracy'] = accuracy
    scores['balanced_accuracy'] = balanced_accuracy
    scores['precision'] = precision
    scores['recall'] = recall
    scores['f1'] = f1
    scores['roc_auc'] = roc_auc

    # Printing average scores
    print(f"Accuracy : {np.mean(accuracy):.2f} ± {np.std(accuracy):.2f}")
    print(f"Balanced accuracy : {np.mean(balanced_accuracy):.2f} ± {np.std(balanced_accuracy):.2f}")
    print(f"Precision : {np.mean(precision):.2f} ± {np.std(precision):.2f}")
    print(f"Recall : {np.mean(recall):.2f} ± {np.std(recall):.2f}")
    print(f"F1-score : {np.mean(f1):.2f} ± {np.std(f1):.2f}")
    print(f"AUC : {np.mean(roc_auc):.3f} ± {np.std(roc_auc):.3f}\n")

    return model, re_x_train


def relabeller(dataframe: pd.DataFrame, y: int):
    data = dataframe.copy()
    data.loc[data['y'] != y, 'y'] = 0
    data.loc[data['y'] == y, 'y'] = 1
    return data


def main():
    # Timer
    start_time = time.time()

    my_kmers = file2list('features_6mer.txt')

    # Reading data
    df = pd.read_parquet('train_6m_4096_v1.parquet')
    test_data = pd.read_parquet('test_6m_4096_v1.parquet')
    train_data = df.drop(columns=['filename'])
    test_data = test_data.drop(columns=['filename'])

    acc, bac, prec, rec, f1l, roc = [], [], [], [], [], []

    # for kmer in my_kmers:
    for kmer in my_kmers:

        tr_data = train_data.copy()

        print(kmer)
        tr_data = tr_data[[kmer, 'y']]

        model = ensemble.RandomForestClassifier(n_jobs=-1)
        trained_model, x_train = cross_val_loop(model, tr_data)

        # Loading test data
        te_data = test_data.copy()

        te_data = te_data[[kmer, 'y']]

        # Splitting test data
        y_test = te_data['y']
        x_test = te_data.drop(columns=['y'])

        x_train, x_test = x_train.align(x_test, join='left', axis=1, fill_value=0)

        # Prediction
        y_pred = trained_model.predict(x_test)
        y_prob = trained_model.predict_proba(x_test)

        # Evaluation
        accuracy = metrics.accuracy_score(y_test, y_pred)
        balanced_accuracy = metrics.balanced_accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred, average='weighted')
        recall = metrics.recall_score(y_test, y_pred, average='weighted')
        f1 = metrics.f1_score(y_test, y_pred, average='weighted')
        roc_auc = metrics.roc_auc_score(y_test, y_prob, average='weighted', multi_class='ovo')

        acc.append(accuracy)
        bac.append(balanced_accuracy)
        prec.append(precision)
        rec.append(recall)
        f1l.append(f1)
        roc.append(roc_auc)

    scores_dict = dict()
    scores_dict['accuracy'] = acc
    scores_dict['balanced_accuracy'] = bac
    scores_dict['precision'] = prec
    scores_dict['recall'] = rec
    scores_dict['f1'] = f1l
    scores_dict['roc_auc'] = roc

    scores_dict['No. of features'] = list(range(1, 7))

    score_df = pd.DataFrame.from_dict(scores_dict)
    score_df.to_csv('6m_scores_seq.csv')

    print('Done')

    end_time = time.time()
    execution_time = end_time - start_time
    print(f'Time taken for execution: {execution_time} seconds.')


if __name__ == "__main__":
    main()
