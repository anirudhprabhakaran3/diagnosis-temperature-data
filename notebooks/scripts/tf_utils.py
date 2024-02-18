import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from scripts.constants import FILTERED_DATA_CSV_PATH

def get_data():
    df = pd.read_csv(FILTERED_DATA_CSV_PATH, index_col=False)
    df = df.sample(frac=1)
    df.reset_index(drop=True, inplace=True)

    X = df.drop("disease", axis=1)
    X = X.drop("disease_encoded", axis=1)

    y = df["disease_encoded"]

    train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)
    train_data = train_data.to_numpy()
    test_data = test_data.to_numpy()
    train_labels = train_labels.to_numpy()
    test_labels =test_labels.to_numpy()

    return train_data, test_data, train_labels, test_labels
