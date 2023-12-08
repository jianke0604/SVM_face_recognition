import cv2
import face_alignment
import numpy as np

import cv2
import face_alignment
from tqdm import tqdm


def cv_draw_landmark(img_ori, pts, box=None, color=(0, 0, 255), size=2):
    img = img_ori.copy()
    n = pts.shape[0]
    for i in range(n):
        cv2.circle(img, (int(round(pts[i, 0])), int(round(pts[i, 1]))), size, color, -1)
    return img


def align(input):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    preds = fa.get_landmarks(input)
    if preds is not None:
        return preds[0]
    else:
        # print('None!')
        # cv2.imshow('No_face_detected', np.array(input, dtype=np.uint8))
        # cv2.waitKey(0)
        return None

def align_data(train_x):
    train_feature = np.zeros((train_x.shape[0], 136))
    for i, img in tqdm(enumerate(train_x)):
        pred = align(img.reshape((48, 48)))
        if pred is not None:
            pred = pred.reshape(136)
        else:
            pred = np.zeros(136)
        train_feature[i] = pred
    return train_feature


if __name__ == '__main__':
    test_img = np.load('train_x_example.npy').reshape((48, 48))
    pred = align(test_img)
    output = cv_draw_landmark(test_img, pred)
    cv2.imshow('result', output)
    cv2.waitKey(0)
