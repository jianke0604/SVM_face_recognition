import os
import sys
import time
import threading
import logging
import torch
import pandas as pd
import argparse
import numpy as np
from model import get_hog_features
if torch.cuda.is_available():
    from thundersvm import SVC
    device = "gpu"
else:
    from sklearn.svm import SVC
    device = "cpu"


def prepare_data(data):
    data = data.groupby("Usage")
    train_x = pd.concat([data.get_group("Training"), data.get_group("PrivateTest")])
    test_x = data.get_group("PublicTest")
    train_y = np.array(train_x["emotion"])
    test_y = np.array(test_x["emotion"])
    train_x = np.array(train_x["pixels"].str.split().tolist(), dtype=int)
    test_x = np.array(test_x["pixels"].str.split().tolist(), dtype=int)
    return train_x, train_y, test_x, test_y


def main(args):
    hog = args.hog
    path = args.path
    print(args)
    print("start loading data")
    data = pd.read_csv(path)
    train_x, train_y, test_x, test_y = prepare_data(data)
    if hog:
        train_x = get_hog_features(train_x)
        test_x = get_hog_features(test_x)
    print("loading data done")
    
    if device == 'cpu':
        print("Using CPU")
        face = SVC(
            C=args.C, kernel=args.kernel, gamma=args.gamma, cache_size=800,
            decision_function_shape='ovr', probability=True,
            random_state=42, verbose=1
        )
    else:
        print("Using GPU")
        face = SVC(
            C=args.C, kernel=args.kernel, gamma=args.gamma, cache_size=800,
            decision_function_shape='ovr', probability=True,
            random_state=42, gpu_id=args.gpu_id, verbose=1
        )

    face.fit(train_x, train_y)
    print(face.score(test_x, test_y))


if '__main__' == __name__:
    parser = argparse.ArgumentParser(description='Description of your script')
    parser.add_argument('-p', '--path', default='./data/fer2013/fer2013.csv', help='Path to the dataset (default: ./data/fer2013/fer2013.csv)')
    parser.add_argument('--hog', default=True, help='Use HOG features (default: True)')
    parser.add_argument('--kernel', default='rbf', help='Kernel type (default: rbf)')
    parser.add_argument('--gamma', default=0.01, help='Kernel coefficient (default: 0.01)')
    parser.add_argument('--C', default=1.0, help='Penalty parameter C of the error term (default: 1.0)')
    parser.add_argument('--gpu_id', type=int, default=0, help='Specify the GPU id (default: 0)')
    
    args = parser.parse_args()
    main(args)
    