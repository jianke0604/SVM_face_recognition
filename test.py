# from thundersvm import SVC
# from sklearn.svm import SVC
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
    # train_x = train_x['pixels'].str.split(' ', expand=True).astype('float32')
    # test_x = test_x['pixels'].str.split(' ', expand=True).astype('float32')
    train_x = get_hog_features(np.array(train_x['pixels'].str.split(' ', expand=True).astype('float32')))
    test_x = get_hog_features(np.array(test_x['pixels'].str.split(' ', expand=True).astype('float32')))
    return train_x, train_y, test_x, test_y

def main(device):
    path = './data/fer2013/fer2013.csv'
    data = pd.read_csv(path)
    print("start loading data")
    train_x, train_y, test_x, test_y = prepare_data(data)
    print("loading data done")
    # print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    
    if device == 'cpu':
        print("Using CPU")
        from sklearn.svm import SVC
    else:
        print("Using GPU")
        from thundersvm import SVC
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        train_x, train_y, test_x, test_y = (
            torch.tensor(train_x).to(device),
            torch.tensor(train_y.values).to(device),
            torch.tensor(test_x).to(device),
            torch.tensor(test_y.values).to(device),
        )
        
    # Create SVM model on GPU
    face = SVC(
        C=1.0, kernel='sigmoid', gamma=0.01, cache_size=800,
        decision_function_shape='ovr', probability=True,
        random_state=42, verbose=1
    )

    # Train and test on GPU
    face.fit(train_x, train_y)
    print(face.score(test_x, test_y))

if '__main__' == __name__:
    parser = argparse.ArgumentParser(description='Description of your script')
    parser.add_argument('-d', '--device', default='cpu', help='Specify the device (default: cpu)')

    args = parser.parse_args()
    main(args.device)
    