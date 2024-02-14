import numpy as np

DISEASE_CLASSES = [
    'dengue',
    'non-infectious diseases',
    'non-tubercular bacterial infection',
    'tuberculosis'
 ]

CLASS_WEIGHTS = np.array([
    47,
    28,
    37,
    32
])

FINAL_DATA_CSV_PATH = "../data/final_data_encoded.csv"
X_TRAIN = "../data/x_train.csv"
X_TEST = "../data/x_test.csv"
Y_TRAIN = "../data/y_train.csv"
Y_TEST = "../data/y_test.csv"


BATCH_SIZE = 8