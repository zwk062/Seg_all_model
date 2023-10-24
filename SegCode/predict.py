import torch
import torch.nn.functional as F
import numpy as np
import os
import glob
import nibabel as nib
from tqdm import tqdm
import SimpleITK as sitk

from utils.train_metrics import metrics3d
# from utils.misc import get_spacing
from skimage import filters
# from utils.vtkpolydataToimage import convertNifti2Metadata
# import vtk
from utils.dice_loss import dice_coeff_loss
from models.Re_mui_net import Re_mui_net

import torch.nn as nn

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch, gc

# DATABASE = ''
#
args = {
    # ## .mha data ##
    # 'test_path': '/media/imed/新加卷/segdata/MRABrain/test',
    # 'pred_path': '/media/imed/新加卷/segdata/MRABrain/4',

    ## .nii data ##
    'test_path': 'D:\\zhangchaoran\\data_301\\test\\',
    'pred_path': 'D:\\zhangchaoran\\data_301\\test\\',

    'img_size': 224
}

if not os.path.exists(args['pred_path']):
    os.makedirs(args['pred_path'])


def standardization_intensity_normalization(dataset, dtype):
    mean = dataset.mean()
    std = dataset.std()
    return ((dataset - mean) / std).astype(dtype)


def rescale(img):
    w, h = img.size
    min_len = min(w, h)
    new_w, new_h = min_len, min_len
    scale_w = (w - new_w) // 2
    scale_h = (h - new_h) // 2
    box = (scale_w, scale_h, scale_w + new_w, scale_h + new_h)
    img = img.crop(box)
    return img


def load_image3d():
    test_images = []
    test_labels = []

    for file in glob.glob(os.path.join(args['test_path'].replace('\\\\', '\\'), 'image', '*.mha')):
        basename = os.path.basename(file)
        # print(file)
        # print(file)
        file_name = basename[:-8]  # for MRABrain
        # file_name = basename[:-10]  # for VascuSynth
        image_name = os.path.join(args['test_path'], 'image', basename)
        # label_name = os.path.join(args['test_path'], 'label', file_name + '-TPGAR-mesh.mha')
        test_images.append(image_name)
        # test_labels.append(label_name)
    return test_images


def load_label3d():
    test_labels = []
    for file in glob.glob(os.path.join(args['pred_path'].replace('\\\\', '\\'), 'label', '*.mha')):
        basename = os.path.basename(file)
        # print(file)
        file_name = basename[:-8]  # for MRABrain
        # file_name = basename[:-10]  # for VascuSynth
        label_name = os.path.join(args['pred_path'], 'label', basename)
        # label_name = os.path.join(args['test_path'], 'label', file_name + '-TPGAR-mesh.mha')
        test_labels.append(label_name)
        # test_labels.append(label_name)
    return test_labels


def load_net():
    # net = torch.load('.//home/imed/disk5TA/hhy/xyx/xyxX-netPatchEnhanced_Dicebest_DSC.pkl', map_location=lambda storage, loc: storage)
    # net = Re_mui_net().cuda()
    net = torch.load('D:/zhangchaoran/NEW_DATA_TRAIN/MRI/our-avg/vnet_Dice2100.pkl')
    # print(net)
    net = nn.DataParallel(net).cuda()
    return net


# def save_prediction(pred, filename='', spacing=None):
#     # pred = torch.argmax(pred, dim=1)
#     save_path = args['pred_path'] + 'pred/'
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#         print("Make dirs success!")
#     # for MSELoss()
#     mask = (pred.data.cpu().numpy() * 255)  # .astype(np.uint8)
#     print(mask.shape)
#     # thresholding
#     threshold = filters.threshold_otsu(mask, nbins=256)
#     # print('threshold',threshold)
#     pred = np.where(mask > threshold, 255.0, 0)
#
#     mask = (pred.squeeze(0)).squeeze(0)  # 3D numpy array
#     # mask = mask.squeeze(0)  # for CE Loss
#     # image = nib.Nifti1Image(np.int32(mask), affine)
#     # nib.save(image, save_path + filename + ".nii.gz")
#     mask = sitk.GetImageFromArray(mask)
#
#     if spacing is not None:
#         mask.SetSpacing(spacing)
#     sitk.WriteImage(mask, os.path.join(save_path + filename + ".nii"))
#     sitk.WriteImage(mask, os.path.join(save_path + filename + ".mha"))
#
#
# def save_label(label, index, spacing=None):
#     label_path = args['pred_path'] + 'label/'
#     if not os.path.exists(label_path):
#         os.makedirs(label_path)
#     # nib.save(label, os.path.join(label_path, index + ".nii.gz"))
#     label = sitk.GetImageFromArray(label)
#     if spacing is not None:
#         label.SetSpacing(spacing)
#     sitk.WriteImage(label, os.path.join(label_path, index + ".mha"))


def predict():
    # print(torch.__version__)
    net = load_net().cuda()
    images = load_image3d()
    labels = load_label3d()
    with torch.no_grad():
        net.eval()
        # print(len(images))
        # print(len(labels))
        ACC, SEN, SPE, IOU, DSC, PRE, AUC = [], [], [], [], [], [], []
        for i in tqdm(range(len(images))):
            # name_list = images[i].split('/')
            # index = name_list[-1][:-4]

            # spacing = get_spacing(images[i])
            # print(i)
            image = sitk.ReadImage(images[i])
            image = sitk.GetArrayFromImage(image).astype(np.float32)
            # print(image)
            Image = standardization_intensity_normalization(image, 'float32')
            # print(labels[i])
            label = sitk.ReadImage(labels[i])
            label = sitk.GetArrayFromImage(label).astype(np.float32)

            # label = standardization_intensity_normalization(label, 'float32')

            # save_label(label, index)

            # if cuda
            image = torch.from_numpy(np.ascontiguousarray(Image)).unsqueeze(0).unsqueeze(0)
            # print(image.shape)
            label = torch.from_numpy(np.ascontiguousarray(label)).unsqueeze(0)
            # label = label / 255
            # print(image.shape)
            # image = image.cuda()
            output = net(image)
            # print(output.shape)
            # loss = dice_coeff_loss(output, label)
            # loss.backward()

            acc, sen, spe, iou, dsc, pre = metrics3d(output, label, output.shape[0])

            print(('acc: {1: .4f}\tsen: {1: .4f}\tspe: {2: .4f}\tiou: {3: .4f}\tdsc: {4: .4f}\tpre: {5: .4f}').format \
                      (acc / output.shape[0], sen / output.shape[0], spe / output.shape[0], iou / output.shape[0],
                       dsc / output.shape[0], pre / output.shape[0]))

            # save_prediction(output, affine=affine, filename=index + '_pred')
            # save_prediction(output, filename=index + '_pred', spacing=spacing)
            ACC.append(acc)
            SEN.append(sen)
            SPE.append(spe)
            IOU.append(iou)
            DSC.append(dsc)
            PRE.append(pre)
    print(('acc: {:.4f}\tsen: {:.4f}\tspe: {:.4f}\tiou: {:.4f}\tdsc: {:.4f}\tpre: {:.4f}').format(
        np.mean(ACC), np.mean(SEN), np.mean(SPE), np.mean(IOU), np.mean(DSC), np.mean(PRE)))


if __name__ == '__main__':
    predict()
