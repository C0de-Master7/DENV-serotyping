from sklearn import model_selection, metrics, ensemble
import numpy as np
import re
import joblib
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd


def cross_val_loop(model, data, splits=10):
    # Reading input data
    y = data['y']
    x = data.drop(columns=['y'])

    # Initialising cross validation data splitter
    skf = model_selection.StratifiedKFold(n_splits=splits, shuffle=True, random_state=7)
    scores = {}
    train_accuracy, train_balanced_accuracy, precision, recall, train_f1, test_f1, roc_auc = [], [], [], [], [], [], []
    test_accuracy, test_balanced_accuracy = [], []
    # re_x_train, re_x_test = [], []

    # Cross validation loop
    for _, (train_index, test_index) in enumerate(skf.split(x, y)):
        # Creating train test split data
        x_train, y_train = x.iloc[train_index], y.iloc[train_index]
        # re_x_train.append(x_train)
        x_test, y_test = x.iloc[test_index], y.iloc[test_index]
        re_x_test = x_test

        # Model training
        model.fit(x_train, y_train)

        # Evaluation
        train_y_pred = model.predict(x_train)
        test_y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)

        # Model evaluation
        test_accuracy.append(metrics.accuracy_score(y_test, test_y_pred))
        test_balanced_accuracy.append(metrics.balanced_accuracy_score(y_test, test_y_pred))
        precision.append(metrics.precision_score(y_test, test_y_pred, average='macro'))
        recall.append(metrics.recall_score(y_test, test_y_pred, average='macro'))
        test_f1.append(metrics.f1_score(y_test, test_y_pred, average='macro'))
        roc_auc.append(metrics.roc_auc_score(y_test, y_prob, average='macro', multi_class='ovr'))

        train_accuracy.append(metrics.accuracy_score(y_train, train_y_pred))
        train_balanced_accuracy.append(metrics.balanced_accuracy_score(y_train, train_y_pred))
        train_f1.append(metrics.f1_score(y_train, train_y_pred, average='macro'))

    # Appending scores
    scores['accuracy'] = test_accuracy
    scores['balanced_accuracy'] = test_balanced_accuracy
    scores['precision'] = precision
    scores['recall'] = recall
    scores['f1'] = test_f1
    scores['roc_auc'] = roc_auc

    # Printing average scores
    print(f"Accuracy : {np.mean(test_accuracy):.2f} ± {np.std(test_accuracy):.2f}")
    print(f"Balanced accuracy : {np.mean(test_balanced_accuracy):.2f} ± {np.std(test_balanced_accuracy):.2f}")
    print(f"Precision : {np.mean(precision):.2f} ± {np.std(precision):.2f}")
    print(f"Recall : {np.mean(recall):.2f} ± {np.std(recall):.2f}")
    print(f"F1-score : {np.mean(test_f1):.2f} ± {np.std(test_f1):.2f}")
    print(f"AUC : {np.mean(roc_auc):.3f} ± {np.std(roc_auc):.3f}\n")

    return (model, train_accuracy, train_balanced_accuracy, train_f1, test_accuracy, test_balanced_accuracy,
            test_f1)


def relabeller(dataframe: pd.DataFrame, y: int):
    data = dataframe.copy()
    data.loc[data['y'] != y, 'y'] = 0
    data.loc[data['y'] == y, 'y'] = 1
    return data


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
            seq = data.iloc[i]['sequence']
            file = data.iloc[i]['filename']
            # y = data.iloc[i]['y']
            temp_dict = {}
            for kmer in kmer_list:
                pattern = re.compile(kmer)
                result = pattern.findall(seq)
                if len(result) != 0:
                    # Append count to dict with seq as key
                    temp_dict[kmer] = len(result)

            # temp_dict['y'] = y
            seq_dict[file] = temp_dict
            print(f"Processed {i + 1} sequence(s)")

        # Converting to dataframe
        data = pd.DataFrame.from_dict(seq_dict, orient='index')
        data.reset_index(level=0, inplace=True)
        data = data.drop(columns=['index'])
        data = data.fillna(0)
        print("Converted to dataframe")
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


def main():
    # Timer
    start_time = time.time()

    train_data = pd.read_parquet('train_6m_data.parquet')
    test_data = pd.read_parquet('test_6m_data.parquet')
    train_data = train_data.drop(columns=['filename'])
    test_data = test_data.drop(columns=['filename'])

    y_test = test_data['y']
    x_test = test_data.drop(columns=['y'])

    model = ensemble.RandomForestClassifier(n_jobs=-1)

    train_data.loc[train_data['y'] == 4, 'y'] = 0
    test_data.loc[test_data['y'] == 4, 'y'] = 0

    (trained_model, train_accuracy, train_balanced_accuracy, train_f1, test_accuracy, test_balanced_accuracy,
     test_f1) = cross_val_loop(model, train_data)

    joblib.dump(trained_model, 'trained_RF_model.pkl')

    # Plot
    plt.plot(range(1, 11), train_f1, label='Training F1')
    plt.plot(range(1, 11), test_f1, label='Validation F1')
    plt.plot(range(1, 11), train_accuracy, label='Training accuracy')
    plt.plot(range(1, 11), test_accuracy, label='Validation accuracy')
    plt.plot(range(1, 11), train_balanced_accuracy, label='Training bal accuracy')
    plt.plot(range(1, 11), test_balanced_accuracy, label='Validation bal accuracy')
    plt.xlabel('Fold')
    plt.ylabel('Mean Squared Error')
    plt.title('Training vs Validation F1 per Fold')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Prediction
    y_pred = trained_model.predict(x_test)
    y_prob = trained_model.predict_proba(x_test)

    # Compute and display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    # Evaluation
    accuracy = metrics.accuracy_score(y_test, y_pred)
    balanced_accuracy = metrics.balanced_accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, average='weighted')
    recall = metrics.recall_score(y_test, y_pred, average='weighted')
    f1 = metrics.f1_score(y_test, y_pred, average='weighted')
    roc_auc = metrics.roc_auc_score(y_test, y_prob, average='weighted', multi_class='ovo')

    # Printing results
    print("Accuracy", accuracy)
    print("Balanced accuracy", balanced_accuracy)
    print("Precision", precision)
    print("Recall", recall)
    print("F1-score", f1)
    print("ROC_AUC", roc_auc, "\n")

    print('Done')

    end_time = time.time()
    execution_time = end_time - start_time
    print(f'Time taken for execution: {execution_time} seconds.')


if __name__ == "__main__":
    main()
