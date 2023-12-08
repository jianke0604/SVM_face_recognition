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

    for i in range(pixels.shape[0]):
        x = pixels[i]
        x = x.reshape((1,) + x.shape)
        j = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir='data/fer2013', save_prefix='aug', save_format='jpeg'):
            j += 1
            if j > 5:
                break

            