import pandas as pd
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator

def prepare_data(data):
    data = data.groupby("Usage")
    train_x = pd.concat([data.get_group("Training"), data.get_group("PrivateTest")])
    test_x = data.get_group("PublicTest")
    train_y = np.array(train_x["emotion"])
    test_y = np.array(test_x["emotion"])
    train_x = np.array(train_x["pixels"].str.split().tolist(), dtype=int)
    test_x = np.array(test_x["pixels"].str.split().tolist(), dtype=int)
    return train_x, train_y, test_x, test_y

def data_argumentation(data):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    dir='E:/BaiduNetdiskDownload/fer2013/train/6'

    for i in range(len(data.shape[0]))
    for batch in datagen.flow(x, batch_size=1,
                            save_to_dir='fer2013TrainDataAdd/6', 
                            save_prefix='6', 
                            save_format='jpeg'):