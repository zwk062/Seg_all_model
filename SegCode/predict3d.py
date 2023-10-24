"""
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃         __  __                           ______          ___                 ┃
┃        / / / /___ _____  ____  __  __   / ____/___  ____/ (_)___  ____ _     ┃
┃       / /_/ / __ `/ __ \/ __ \/ / / /  / /   / __ \/ __  / / __ \/ __ `/     ┃
┃      / __  / /_/ / /_/ / /_/ / /_/ /  / /___/ /_/ / /_/ / / / / / /_/ /      ┃
┃     /_/ /_/\__,_/ .___/ .___/\__, /   \____/\____/\__,_/_/_/ /_/\__, /       ┃
┃                /_/   /_/    /____/                             /____/        ┃
┃                                                                              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import numpy as np
import os
import glob
import nibabel as nib
from tqdm import tqdm
import SimpleITK as sitk
from utils.misc import get_spacing
from skimage import filters
#from utils.vtkpolydataToimage import convertNifti2Metadata
from utils.losses import WeightedCrossEntropyLoss, dice_coeff_loss
import vtk
import torch.nn as nn
from models.Re_mui_net import Re_mui_net
from utils.train_metrics import metrics3d

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

#DATABASE = ''
#
args = {
    # ## .mha data ##
    # 'test_path': '/media/imed/新加卷/segdata/MRABrain/test',
    # 'pred_path': '/media/imed/新加卷/segdata/MRABrain/4',

    ## .nii data ##
    'test_path': 'D:\\zhangchaoran\\data_301\\test',
    'pred_path': 'D:\\zhangchaoran\\data_301\\test',

    # 'img_size': 224
}

if not os.path.exists(args['pred_path']):
    os.makedirs(args['pred_path'])

def standardization_intensity_normalization(dataset, dtype):
    mean = dataset.mean()
    std  = dataset.std()
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


def load_3d():
    test_images = []
    test_labels = []
    for file in glob.glob(os.path.join(args['test_path'], 'image', '*.mha')):
        basename = os.path.basename(file)
        print (file)
        file_name = basename[:-8]  # for MRABrain
        #file_name = basename[:-10]  # for VascuSynth
        image_name = os.path.join(args['test_path'], 'image', basename)
        #label_name = os.path.join(args['test_path'], 'label', file_name + '-TPGAR-mesh.mha')
        test_images.append(image_name)
        #test_labels.append(label_name)
    return test_images, test_labels
def load_image3d():
    test_images = []
    test_labels = []
    for file in glob.glob(os.path.join(args['test_path'], 'image', '*.mha')):
        basename = os.path.basename(file)
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
    for file in glob.glob(os.path.join(args['pred_path'], 'label', '*.mha')):
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
    net = torch.load('D:/zhangchaoran/NEW_DATA_TRAIN/public train/our net/train/vnet_Dicebest_DSC.pkl')
    # all_layers =list(net.children())
    # net = all_layers[-2]
    # print(net)
    # print(net)
    # pre_dict = net.state_dict()  # 按键值对将模型参数加载到pre_dict

    # for k, v in pre_dict.items():  # 打印模型参数
    #
    #     print(k)  # 打印模型每层命名
    net = nn.DataParallel(net).cuda()

    return net



def save_prediction(pred, filename='', spacing=None):
    # pred = torch.argmax(pred, dim=1)
    save_path = args['pred_path'] + 'pred/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("Make dirs success!")
    # for MSELoss()
    mask = (pred.data.cpu().numpy() * 255)  # .astype(np.uint8)
    print(mask.shape)
            # thresholding
    threshold = filters.threshold_otsu(mask, nbins=256)
    # print('threshold',threshold)
    pred = np.where(mask > threshold, 255.0, 0)



    mask = (pred.squeeze(0)).squeeze(0)  # 3D numpy array
    # mask = mask.squeeze(0)  # for CE Loss
    # image = nib.Nifti1Image(np.int32(mask), affine)
    # nib.save(image, save_path + filename + ".nii.gz")
    mask = sitk.GetImageFromArray(mask)

    if spacing is not None:
        mask.SetSpacing(spacing)
    # sitk.WriteImage(mask, os.path.join(save_path + filename + ".nii"))
    sitk.WriteImage(mask, os.path.join(save_path, filename + ".mha"))


def save_label(label, index, spacing=None):
    label_path = args['pred_path'] + 'label/'
    if not os.path.exists(label_path):
        os.makedirs(label_path)
    # nib.save(label, os.path.join(label_path, index + ".nii.gz"))
    label = sitk.GetImageFromArray(label)
    if spacing is not None:
        label.SetSpacing(spacing)
    sitk.WriteImage(label, os.path.join(label_path, index + ".mha"))


def predict():
    net = load_net()
    images = load_image3d()


    # label = load_label3d()
    with torch.no_grad():
        net.eval()
        for i in tqdm(range(len(images))):
            name_list = images[i].split('/')
            index = name_list[-1][:-4]

            spacing = get_spacing(images[i])

            image = sitk.ReadImage(images[i])
            image = sitk.GetArrayFromImage(image).astype(np.float32)
            Image = standardization_intensity_normalization(image, 'float32')
            #
            # label = sitk.ReadImage(labels[i])
            # label = sitk.GetArrayFromImage(label).astype(np.float32)
            # label = label / 255
            # loss = dice_coeff_loss(Image, label)
            # loss.backward()

            # auc, acc, sen, spe, iou, dsc, pre = metrics3d(Image, label,1)
            # print(
            #     '{0:d}:{1:d}] \u2501\u2501\u2501 loss:{2:.10f}\tacc:{3:.4f}\tsen:{4:.4f}\tspe:{5:.4f}\tiou:{6:.4f}\tdsc:{7:.4f}\tpre:{8:.4f}'.format
            #     (epoch + 1, iters, loss.item(), acc / Image.shape[0], sen / Image.shape[0], spe / Image.shape[0],
            #      iou / Image.shape[0], dsc / Image.shape[0], pre / Image.shape[0]))
            # save_label(label, index)

            # if cuda
            image = torch.from_numpy(np.ascontiguousarray(Image)).unsqueeze(0).unsqueeze(0)
            print(image.shape)
            image = image.cuda()
            output = net(image)

            # save_prediction(output, affine=affine, filename=index + '_pred')
            save_prediction(output, filename=index + '_pred', spacing=spacing)


if __name__ == '__main__':
    

    predict()

    # reader = vtk.vtkMetaImageReader()
    # reader.SetFileName("/home/leila/Desktop/BrainVessel/Normal001-MRA.mha")
    # reader.Update()
    # origin = reader.GetOutput().GetOrigin()
    #convertNifti2Metadata(args['pred_path'] + "pred/", args['pred_path'] + "mha_files/", origin)
