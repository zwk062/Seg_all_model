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
# import torch
# import torch.nn.functional as F
import numpy as np
import os
import glob
import nibabel as nib
from tqdm import tqdm
import SimpleITK as sitk
from utils.misc import get_spacing
from skimage import filters
#from utils.vtkpolydataToimage import convertNifti2Metadata
import vtk
# import torch.nn as nn

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

#DATABASE = ''
#
args = {
    # ## .mha data ##
    'test_path': '/home/imed/segmentation/evaluate/evaluation_seg/1/t',
    'save_path': '/home/imed/segmentation/evaluate/evaluation_seg/1/t/2',

    ## .nii data ##
    # 'test_path': '/media/imed/新加卷/segdata/Bullitt_Isotrope_light_mult32/split/test',
    # 'pred_path': '/media/imed/新加卷/segdata/Bullitt_Isotrope_light_mult32/split/1',

    'img_size': 224
}

if not os.path.exists(args['save_path']):
    os.makedirs(args['save_path'])

# def standardization_intensity_normalization(dataset, dtype):
#     mean = dataset.mean()
#     std  = dataset.std()
#     return ((dataset - mean) / std).astype(dtype)
# def rescale(img):
#     w, h = img.size
#     min_len = min(w, h)
#     new_w, new_h = min_len, min_len
#     scale_w = (w - new_w) // 2
#     scale_h = (h - new_h) // 2
#     box = (scale_w, scale_h, scale_w + new_w, scale_h + new_h)
#     img = img.crop(box)
#     return img


# def load_3d():
#     test_images = []
#     test_labels = []
#     for file in glob.glob(os.path.join(args['test_path'], 'image', '*.mha')):
#         basename = os.path.basename(file)
#         print (file)
#         file_name = basename[:-8]  # for MRABrain
#         #file_name = basename[:-10]  # for VascuSynth
#         image_name = os.path.join(args['test_path'], 'image', basename)
#         #label_name = os.path.join(args['test_path'], 'label', file_name + '-TPGAR-mesh.mha')
#         test_images.append(image_name)
#         #test_labels.append(label_name)
#     return test_images, test_labels


# def load_net():
#     # net = torch.load('.//home/imed/disk5TA/hhy/xyx/xyxX-netPatchEnhanced_Dicebest_DSC.pkl', map_location=lambda storage, loc: storage)
#     net = torch.load('/media/imed/新加卷/segdata/cerebravascular/v-NetoraldataDice/v-netoraldata_Dice2300.pkl')
#     # print(net)
#     net = nn.DataParallel(net).cuda()
#     return net


# def save_prediction(pred, filename='', spacing=None):
#     # pred = torch.argmax(pred, dim=1)
#     save_path = args['pred_path'] + 'pred/'
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#         print("Make dirs success!")
#     # for MSELoss()
#     mask = (pred.data.cpu().numpy() * 255)  # .astype(np.uint8)
#     print(mask.shape)
#     #         # thresholding
#     # threshold = filters.threshold_otsu(mask, nbins=256)
#     # # print('threshold',threshold)
#     # pred = np.where(mask > threshold, 255.0, 0)



#     # mask = (pred.squeeze(0)).squeeze(0)  # 3D numpy array
    
#     mask = sitk.GetImageFromArray(mask)

#     if spacing is not None:
#         mask.SetSpacing(spacing)
#     sitk.WriteImage(mask, os.path.join(save_path + filename + ".nii"))
#     sitk.WriteImage(mask, os.path.join(save_path + filename + ".mha"))


# def save_label(label, index, spacing=None):
#     label_path = args['pred_path'] + 'label/'
#     if not os.path.exists(label_path):
#         os.makedirs(label_path)
#     # nib.save(label, os.path.join(label_path, index + ".nii.gz"))
#     label = sitk.GetImageFromArray(label)
#     if spacing is not None:
#         label.SetSpacing(spacing)
#     sitk.WriteImage(label, os.path.join(label_path, index + ".mha"))


# def predict():
#     # net = load_net()
#     images = load_3d()
#     with torch.no_grad():
#         # net.eval()
#         for i in tqdm(range(len(images))):
#             name_list = images[i].split('/')
#             index = name_list[-1][:-4]

#             spacing = get_spacing(images[i])

#             image = sitk.ReadImage(images[i])
#             image = sitk.GetArrayFromImage(image).astype(np.float32)
#             # Image = standardization_intensity_normalization(image, 'float32')

#             # label = sitk.ReadImage(labels[i])
#             # label = sitk.GetArrayFromImage(label).astype(np.float32)
#             # label = label / 255

#             #save_label(label, index)

#             # if cuda
#             # output = torch.from_numpy(np.ascontiguousarray(Image)).unsqueeze(0).unsqueeze(0)
#             output = torch.from_numpy(np.ascontiguousarray(image))
#             # print(image.shape)
#             # image = image.cuda()
#             # output = net(image)

#             # save_prediction(output, affine=affine, filename=index + '_pred')
#             save_prediction(output, filename=index + '_pred', spacing=spacing)


# if __name__ == '__main__':
    

#     predict()

    # reader = vtk.vtkMetaImageReader()
    # reader.SetFileName("/home/leila/Desktop/BrainVessel/Normal001-MRA.mha")
    # reader.Update()
    # origin = reader.GetOutput().GetOrigin()
    #convertNifti2Metadata(args['pred_path'] + "pred/", args['pred_path'] + "mha_files/", origin)


# test_images = []
# # for file in glob.glob(os.path.join(args['test_path'], 't', '*.mha')):
# for file in glob.glob(os.path.join('/home/imed/segmentation/evaluate/evaluation_seg/1/t', '*.mha')):
#     basename = os.path.basename(file)
#     print (file)
#     file_name = basename[:-8]  # for MRABrain
    
#     # image_name = os.path.join(args['test_path'], 't', basename)
#     image_name = os.path.join('/home/imed/segmentation/evaluate/evaluation_seg/1/t', basename)
#     images = test_images.append(image_name)


#     for i in tqdm(range(len(images))):
#         name_list = images[i].split('/')
#         index = name_list[-1][:-4]

#         # spacing = get_spacing(images[i])

#         image = sitk.ReadImage(images[i])
#         image = sitk.GetArrayFromImage(image).astype(np.float32)
#         mask = (image.data.cpu().numpy() * 255)  # .astype(np.uint8)

#         mask = sitk.GetImageFromArray(mask)

#         # if spacing is not None:
#         #     mask.SetSpacing(spacing)
#         sitk.WriteImage(mask, os.path.join(save_path + filename + ".mha"))
image = sitk.ReadImage('/home/imed/segmentation/evaluate/evaluation_seg/1/t/Normal086_pred.mha')
image = sitk.GetArrayFromImage(image).astype(np.float32)
# print(image.max(), image.min())
# mask = (image.numpy() * 255)
mask = image * 255  # .astype(np.uint8)
mask = sitk.GetImageFromArray(mask)
sitk.WriteImage(mask, '/home/imed/segmentation/evaluate/evaluation_seg/1/t/086.mha')