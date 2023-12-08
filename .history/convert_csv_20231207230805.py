import os
import numpy as np
from PIL import Image
import pandas as pd

def images_to_csv(image_dir, csv_file):
    image_files = os.listdir(image_dir)
    data = []
    for image_file in image_files:
        image = Image.open(os.path.join(image_dir, image_file))
        image_array = np.array(image)
        image_array = image_array.reshape(-1)
        image_str = ' '.join(map(str, image_array))
        data.append(image_str)
    df = pd.DataFrame(data, columns=['pixels'])
    df.to_csv(csv_file, index=False)

images_to_csv('data/fer2013', 'data/fer2013/augmented.csv')