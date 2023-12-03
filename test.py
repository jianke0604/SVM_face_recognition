import torch
import pandas as pd
import argparse
import numpy as np
from model import get_hog_features


def prepare_data(data):
    train_x = data[data['Usage'] != 'PublicTest']
    test_x = data[data['Usage'] == 'PublicTest']
    train_y = train_x['emotion']
    test_y = test_x['emotion']
    train_x = train_x['pixels'].str.split(' ', expand=True).astype('float32')
    test_x = test_x['pixels'].str.split(' ', expand=True).astype('float32')
    return train_x, train_y, test_x, test_y

def main(args):
    device = args.device
    hog = args.hog
    path = args.path
    data = pd.read_csv(path)
    print(args)
    print("start loading data")
    train_x, train_y, test_x, test_y = prepare_data(data)
    if hog:
        train_x = get_hog_features(np.array(train_x))
        test_x = get_hog_features(np.array(test_x))
    print("loading data done")
    # print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    
    if device == 'cpu':
        print("Using CPU")
        from sklearn.svm import SVC
        face = SVC(
            C=args.C, kernel=args.kernel, gamma=args.gamma, cache_size=800,
            decision_function_shape='ovr', probability=True,
            random_state=42, verbose=1
        )
    else:
        print("Using GPU")
        from thundersvm import SVC
        
        # Create SVM model on GPU
        face = SVC(
            C=args.C, kernel=args.kernel, gamma=args.gamma, cache_size=800,
            decision_function_shape='ovr', probability=True,
            random_state=42, gpu_id=args.gpu_id, verbose=1
        )

    # Train and test on GPU
    face.fit(train_x, train_y)
    print(face.score(test_x, test_y))

if '__main__' == __name__:
    parser = argparse.ArgumentParser(description='Description of your script')
    parser.add_argument('-p', '--path', default='./data/fer2013/fer2013.csv', help='Path to the dataset (default: ./data/fer2013/fer2013.csv)')
    parser.add_argument('-d', '--device', default='cpu', help='Specify the device (default: cpu)')
    parser.add_argument('--hog', default=True, help='Use HOG features (default: True)')
    parser.add_argument('--kernel', default='rbf', help='Kernel type (default: rbf)')
    parser.add_argument('--gamma', default=0.01, help='Kernel coefficient (default: 0.01)')
    parser.add_argument('--C', default=1.0, help='Penalty parameter C of the error term (default: 1.0)')
    parser.add_argument('--gpu_id', type=int ,default=1, help='Specify the GPU id (default: 1)')

    args = parser.parse_args()
    main(args)
    