import os
import numpy as np
from PIL import Image
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

def data_argumentation():
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    dir='data/fer2013/fer2013.csv'
    data = pd.read_csv(dir)
    pixels = data['pixels'].apply(lambda x: np.array(x.split(), dtype='float32'))
    pixels = np.array(pixels.tolist()).reshape(-1, 48, 48, 1)
    print(pixels.shape)

    for i in range(pixels.shape[0]):
        x = pixels[i]
        x = x.reshape((1,) + x.shape)
        j = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir='data/fer2013/images', save_prefix=f"aug_{data['emotion'][i]}_{data['Usage'][i]}", save_format='jpeg'):
            j += 1
            if j > 5:
                break

def images_to_csv(image_dir, csv_file):
    image_files = os.listdir(image_dir)
    data = []
    for image_file in image_files:
        if 'aug' in image_file:
            emotion, usage = image_file.split('_')[1:3]
            image = Image.open(os.path.join(image_dir, image_file))
            image_array = np.array(image)
            image_array = image_array.reshape(-1)
            image_str = ' '.join(map(str, image_array))
            data.append([emotion, image_str, usage])
    df = pd.DataFrame(data, columns=['emotion', 'pixels', 'Usage'])
    df.to_csv(csv_file, index=False)

if __name__ == '__main__':
    data_argumentation()
    images_to_csv('data/fer2013/images', 'data/fer2013/augmented.csv')