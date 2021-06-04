import numpy as np


def get_accu(y, y_pred):
    return np.sum(y_pred == y) / len(y) * 100
