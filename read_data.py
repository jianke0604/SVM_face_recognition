import pandas as pd
import numpy as np


def read_data(data):
    data = data.groupby("Usage")
    train_df = pd.concat([data.get_group("Training"), data.get_group("PrivateTest")])
    train_y = np.array(train_df["emotion"])
    train_x = np.array(train_df["pixels"].str.split().tolist(), dtype=int)
    test_df = data.get_group("PublicTest")
    test_y = np.array(test_df["emotion"])
    test_x = np.array(test_df["pixels"].str.split().tolist(), dtype=int)
    
    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
    read_data(pd.read_csv('data/fer2013/fer2013.csv'))