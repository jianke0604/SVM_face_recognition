import pandas as pd
import numpy as np
from data_augmentation import data_augmentation
from collections import Counter


def prepare_data(data, test_aug):
    data = data.groupby("Usage")
    train_x = pd.concat([data.get_group("Training"), data.get_group("PrivateTest")])
    train_y = np.array(train_x["emotion"])
    train_x = np.array(train_x["pixels"].str.split().tolist(), dtype=int)
    test_x = data.get_group("PublicTest")
    test_y = np.array(test_x["emotion"])
    if test_aug:
        test_x = test_x.reset_index(drop=True)
        test_x = data_augmentation(test_x)
        print()
    else:
        test_x = np.array(test_x["pixels"].str.split().tolist(), dtype=int)
    return train_x, train_y, test_x, test_y

def concat_and_replace(arr1, arr2, arr3, arr4, arr5):
    concatenated_array = np.vstack((arr1, arr2, arr3, arr4, arr5))
    most_common_elements = np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=concatenated_array)
    return most_common_elements

if __name__ == '__main__':
    prepare_data(pd.read_csv('data/fer2013/fer2013.csv'), test_aug=True)