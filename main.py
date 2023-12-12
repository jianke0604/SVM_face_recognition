import cv2
import numpy as np
import torch
import pandas as pd
import argparse

from model.hog import get_hog_features
from model.resnet import ResNet
from read_data import read_data
from sklearn.decomposition import PCA
from face_align import align_data


def main(args):
    print(args)
    if torch.cuda.is_available() and args.device == "gpu":
        from thundersvm import SVC
        device = "cuda"
    else:
        from sklearn.svm import SVC
        device = "cpu"
    method = args.method
    path = args.path
    pca = args.pca

    print("start loading data")
    data = pd.read_csv(path)
    train_x, train_y, test_x, test_y = read_data(data)


    if method == "cnn":  # cnn
        train_x = torch.tensor(train_x / 255.0).view(-1, 1, 48, 48).to(torch.float32).to(device)
        test_x = torch.tensor(test_x / 255.0).view(-1, 1, 48, 48).to(torch.float32).to(device)
        print(train_x.shape)  # torch.Size([32298, 1, 48, 48])
        print(test_x.shape)   # torch.Size([3589, 1, 48, 48])
        model = ResNet().to(device)
        state_dict = torch.load("resnet_pretrained/epoch19.pth", map_location=device)
        state_dict = {k: v for k, v in state_dict.items() if k != "backbone.fc.weight" and k != "backbone.fc.bias"}
        model.load_state_dict(state_dict)
        model.eval()
        with torch.no_grad():
            # train_x_list = torch.split(train_x, train_x.shape[0] / 4 + 1, dim=0)
            # output = []
            # for x in train_x_list:
            #     output.append(model(x).detach().cpu())
            # train_x = torch.cat(output, dim=0)
            train_x = model(train_x).detach().cpu().tolist()
            test_x = model(test_x).detach().cpu().tolist()
        # print(len(train_x), len(train_x[0]))  # 32298 512
        # print(len(test_x), len(test_x[0]))    # 3589 512
        # print(train_x.shape)  # torch.Size([32298, 512])
        # print(test_x.shape)   # torch.Size([3589, 512])
    elif method == "hog":
        train_x = get_hog_features(train_x)
        test_x = get_hog_features(test_x)
        print(len(train_x), len(train_x[0]))  # 32298 900
        print(len(test_x), len(test_x[0]))    # 3589 900
        # print(train_x.shape, train_x.device, type(train_x))
        # print(test_x.shape, test_x.device, type(test_x))
        if pca:
            pca = PCA(n_components=args.nComponents)
            train_x = pca.fit_transform(train_x)
            test_x = pca.transform(test_x)
            # test_x = pca.fit_transform(test_x)
            # train_x = pca.transform(train_x)
    elif method == "alignment":
        train_x = align_data(train_x)
        test_x = align_data(test_x)


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
    print("training is done.")
    print(face.score(test_x, test_y))
    print(face.predict(test_x).shape)
    print(face.predict(test_x)[:100])


if '__main__' == __name__:
    parser = argparse.ArgumentParser(description='Description of your script')
    parser.add_argument('-p', '--path', default='./data/fer2013/fer2013.csv', help='Path to the dataset (default: ./data/fer2013/fer2013.csv)')
    parser.add_argument('--method', default='cnn', help='Method to extract features (default: "cnn")')
    parser.add_argument('--kernel', default='rbf', help='Kernel type (default: rbf)')
    parser.add_argument('--gamma', type=float, default=0.01, help='Kernel coefficient (default: 0.01)')
    parser.add_argument('--C', type=float, default=1.0, help='Penalty parameter C of the error term (default: 1.0)')
    parser.add_argument('--device', default='gpu', help='Device to use (default: gpu)')
    parser.add_argument('--gpu_id', type=int, default=0, help='Specify the GPU id (default: 0)')
    parser.add_argument('--pca', default=False)
    parser.add_argument('--nComponents', default=1296, help='Specify the feature number of PCA')

    args = parser.parse_args()
    main(args)
    