import cv2
from skimage.feature import hog


def get_hog_features(samples):
    features = []
    for sample in samples:
        sample = sample.reshape(48, 48)
        hog_features = hog(sample, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        features.append(hog_features)
    return features