import os
import numpy as np
from PIL import Image
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

def data_augmentation(data_df = None, AUG_NUM = 5):
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    if data_df is None:
        dir='data/fer2013/fer2013.csv'
        data_df = pd.read_csv(dir)
    pixels = data_df['pixels'].apply(lambda x: np.array(x.split(), dtype='float32'))
    pixels = np.array(pixels.tolist()).reshape(-1, 48, 48, 1)
    print('input data shapes: ', pixels.shape)
    image_dir = 'data/fer2013/images'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    # csv_file = 'data/fer2013/augmented.csv'
    data_list = [[] for _ in range(AUG_NUM)]

    for i in range(pixels.shape[0]):
        x = pixels[i]
        x = x.reshape((1,) + x.shape)
        # Save the original image
        Image.fromarray(x[0].squeeze()).convert('L').save(f"data/fer2013/images/aug_{data_df['emotion'][i]}_{data_df['Usage'][i]}_origin.jpeg")
        if data_df['Usage'][i] != 'PublicTest':
            print(f"Skip the image {i} not in PublicTest")
            pass
        else:
            for batch in datagen.flow(x, batch_size=1, save_to_dir='data/fer2013/images/', save_prefix=f"aug_{data_df['emotion'][i]}_{data_df['Usage'][i]}", save_format='jpeg'):
                image_files = os.listdir(image_dir)
                if len(image_files) == AUG_NUM:
                    break
        image_files = os.listdir(image_dir)
        assert len(image_files) == AUG_NUM, f"The number of augmented images {i} is {len(image_files)}, should be {AUG_NUM}."

        
        for j, image_file in enumerate(image_files):
            if 'aug' in image_file:
                emotion, usage = image_file.split('_')[1:3]
                image = Image.open(os.path.join(image_dir, image_file))
                image_array = np.array(image)
                image_array = image_array.reshape(-1)
                image_str = ' '.join(map(str, image_array))
                data_list[j % 5].append([emotion, image_str, usage])
                # os.remove(os.path.join(image_dir, image_file))  # Delete the image file after processing
        # assert len(data_list) == 5 * (i + 1), f"The number of augmented images is {len(data_list)}, should be {5 * (i + 1)}"
        for image_file in image_files:
            if 'aug' in image_file:
                os.remove(os.path.join(image_dir, image_file))
        print_progress_bar(i + 1, pixels.shape[0])
    for i in range(AUG_NUM):
        print(f"\nTotal number of augmented images in group {i}: {len(data_list[i])}")
    # dfs = [pd.DataFrame(data_list[i], columns=['emotion', 'pixels', 'Usage']) for i in range(AUG_NUM)]
    dfs = [pd.DataFrame(data_list[i], columns=['emotion', 'pixels', 'Usage']) for i in range(AUG_NUM)]
    np_arrays = [np.array(df["pixels"].str.split().tolist(), dtype=int) for df in dfs]
    # df.to_csv(csv_file, index=False)
    return np_arrays

def print_progress_bar(current, total):
    progress = int((current / total) * 10)
    progress_bar = '#' * progress + '-' * (10 - progress)
    print(f"\r{progress_bar} {current}/{total}", end='')

if __name__ == '__main__':
    data_augmentation()