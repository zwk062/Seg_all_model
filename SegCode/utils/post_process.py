import cv2
import os
import glob
from skimage import exposure, morphology, color


def get_centerline(path):
    for file in glob.glob(os.path.join(path, '*otsu.png')):
        index = os.path.basename(file)[:-4]
        image = cv2.imread(file, flags=-1)
        image = image / 255
        skelton = morphology.skeletonize(image)
        save_name = path + index + '_skelton.png'
        cv2.imwrite(save_name, skelton * 255)
        print(file, '\tdone!')


if __name__ == '__main__':
    path = '/home/leila/PycharmProjects/Attention/assets/NERVE/pred/'
    get_centerline(path)
