import numpy as np
import os
import glob
import cv2
import torch.nn as nn
import torch
from PIL import ImageOps, Image
from sklearn.metrics import confusion_matrix
import SimpleITK as sitk
import tqdm
import vtk


def threshold(image):
    image[image >= 100] = 255
    image[image < 100] = 0
    return image


def ReScaleSize(image, re_size=512):
    w, h = image.size
    max_len = max(w, h)
    new_w, new_h = max_len, max_len
    delta_w = new_w - w
    delta_h = new_h - h
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    image = ImageOps.expand(image, padding, fill=0)
    # origin_w, origin_h = w, h
    image = image.resize((re_size, re_size))
    return image  # , origin_w, origin_h


def restore_origin_size(image, origin_w, origin_h):
    max_len = max(origin_w, origin_h)
    image = image.resize((max_len, max_len))
    w, h = image.size
    delta_w, delta_h = (max_len - origin_w) / 2, (max_len - origin_h) / 2
    box = (delta_w, delta_h, w - delta_w, h - delta_h)
    image = image.crop(box)
    return image


def Crop(image):
    left = 261
    top = 1
    right = 1110
    bottom = 850
    image = image.crop((left, top, right, bottom))
    return image


def thresh_OTSU(path):
    for file in glob.glob(os.path.join(path, '*pred.png')):
        index = os.path.basename(file)[:-4]
        image = cv2.imread(file)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        thresh, img = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        # cv2.imwrite(os.path.join(path, index + '_otsu.png'), img)
        cv2.imwrite(file, img)
        print(file, '\tdone!')


def RGB2Gray(image):
    image = image.convert("L")
    return image


def center_crop(image, label):
    center_x, center_y = image.size
    center_x = center_x // 2
    center_y = center_y // 2
    left = center_x - 184
    top = center_y - 184
    right = center_x + 184
    bottom = center_y + 184
    box = (left, top, right, bottom)
    image = image.crop(box)
    label = label.crop(box)
    return image, label


def get_spacing(file):
    import vtk

    reader = vtk.vtkMetaImageReader()
    reader.SetFileName(file)
    reader.Update()
    spacing = reader.GetOutput().GetSpacing()

    return spacing


def get_class_weights(path):
    class_0 = []
    class_1 = []
    for file in glob.glob(os.path.join(path, "training", "mesh_label", "*.mha")):
        img = sitk.ReadImage(file)
        data = sitk.GetArrayFromImage(img)
        num_0 = np.sum(data == 0)
        num_1 = np.sum(data > 0)
        max_num = max(num_0, num_1)
        weight_0 = max_num / num_0
        weight_1 = max_num / num_1
        class_0.append(weight_0)
        class_1.append(weight_1)
    class_weight_0 = np.mean(class_0)
    class_weight_1 = np.mean(class_1)
    return [class_weight_0, class_weight_1]


def expand(path, save_path):
    for file in tqdm.tqdm(glob.glob(os.path.join(path, "*.mha"))):
        reader = vtk.vtkMetaImageReader()
        reader.SetFileName(file)
        reader.Update()
        metadata = reader.GetOutput()
        spacing = metadata.GetSpacing()
        image = sitk.ReadImage(file)
        data = sitk.GetArrayFromImage(image)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

        for i in range(data.shape[0]):
            img = data[i, :, :]
            img = cv2.dilate(img, kernel)
            data[i, :, :] = img

        image = sitk.GetImageFromArray(data)
        image.SetSpacing(spacing)
        sitk.WriteImage(image, os.path.join(save_path, os.path.basename(file)[:-4] + "-dilate.mha"))
