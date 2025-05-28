import joblib
import shap
import pandas as pd
import numpy as np


def main():
    train_data = pd.read_parquet('train_6m_data.parquet')
    test_data = pd.read_parquet('test_6m_data.parquet')
    train_data = train_data.drop(columns=['filename'])
    test_data = test_data.drop(columns=['filename'])

    y_train = train_data['y']
    x_train = train_data.drop(columns=['y'])

    y_test = test_data['y']
    x_test = test_data.drop(columns=['y'])

    x_train, x_test = x_train.align(x_test, join='left', axis=1, fill_value=0)

    trained_model = joblib.load('trained_RF_model.pkl')

    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(trained_model)
    shap_values = explainer(x_test)

    # get model predictions
    preds = trained_model.predict(x_test)

    # Aggregate mean shap values for predicted class
    new_shap_values = []
    for i, pred in enumerate(preds):
        # get shap values for predicted class
        new_shap_values.append(shap_values.values[i][:, pred])

    # replace shap values
    shap_values.values = np.array(new_shap_values)
    print(shap_values.shape)

    shap.plots.bar(shap_values, max_display=15)
    shap.plots.beeswarm(shap_values, max_display=15)


if __name__ == "__main__":
    main()
