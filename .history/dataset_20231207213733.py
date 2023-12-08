import pandas as pd
import numpy as np


def prepare_data(data):
    data = data.groupby("Usage")
    train_x = pd.concat([data.get_group("Training"), data.get_group("PrivateTest")])
    test_x = data.get_group("PublicTest")
    train_y = np.array(train_x["emotion"])
    test_y = np.array(test_x["emotion"])
    train_x = np.array(train_x["pixels"].str.split().tolist(), dtype=int)
    test_x = np.array(test_x["pixels"].str.split().tolist(), dtype=int)
    return train_x, train_y, test_x, test_y

def