import pandas as pd
import numpy as np
from torchvision.transforms import transforms
import torch
from PIL import Image
import random

class RandomApplyWithProb:
    def __init__(self, transform, p=0.5):
        self.transform = transform
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return self.transform(img)
        return img

# Define the interpolation method
interpolation = transforms.InterpolationMode.BILINEAR


def prepare_data(data):
    data = data.groupby("Usage")
    train_x = pd.concat([data.get_group("Training"), data.get_group("PrivateTest")])
    train_y = np.array(train_x["emotion"])
    train_x = np.array(train_x["pixels"].str.split().tolist(), dtype=int)
    test_x = data.get_group("PublicTest")
    test_y = np.array(test_x["emotion"])
    test_x = np.array(test_x["pixels"].str.split().tolist(), dtype=int)
    original_train_x = train_x.copy()
    mean = train_x.mean()
    std = train_x.std()

    # parameters
    # Rotation: 15, 0.5
    # Resize: 0.1, 0.5
    # HorizontalFlip: 0.5
    # Shift: (0.1, 0.1), 0.5
    # Shear: 10, 0.5
    train_transform = transforms.Compose([
        RandomApplyWithProb(transforms.RandomRotation(15), p=0.5),
        RandomApplyWithProb(transforms.RandomResizedCrop(size=(48, 48), scale=(0.9, 1.1), ratio=(0.9, 1.1)), p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        RandomApplyWithProb(transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=3), p=0.5),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=mean, std=std)
    ])
    train_x = [train_transform(Image.fromarray(x.reshape(48, 48).astype(np.uint8))) for x in train_x]
    train_x = np.array(torch.stack(train_x)).reshape((-1, 2304))
    
    test_x = (test_x - mean) / std
    
    train_x = np.concatenate((original_train_x, train_x), axis=0)
    train_y = np.concatenate((train_y, train_y), axis=0)

    # train_transform = transforms.Compose([
    #     transforms.RandomHorizontalFlip(hflip_prob),
    #     autoaugment.TrivialAugmentWide(interpolation=interpolation),
    #     transforms.PILToTensor(),
    #     transforms.ConvertImageDtype(torch.float),
    #     transforms.Normalize(mean=mean, std=std)
    # ])

    # if test_aug:
    #     test_x = test_x.reset_index(drop=True)
    #     test_x = data_augmentation(test_x)
    # else:
    #     test_x = np.array(test_x["pixels"].str.split().tolist(), dtype=int)
    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
    prepare_data(pd.read_csv('data/fer2013/fer2013.csv'))