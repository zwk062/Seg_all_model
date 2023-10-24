# import torch.utils.data as data
# import numpy as np
# import os
# from PIL import Image
# from torchvision import transforms
# import SimpleITK as sitk
# import nibabel as nib
# import torch


# class MRADataset(data.Dataset):

#     def __init__(self, root, isTraining=True):
#         self.labelPath = self.get_dataPath(root, False)
#         self.imgPath = root
#         self.isTraining = isTraining
#         self.name = ' '
#         self.root = root

#     def standardization_intensity_normalization(self,dataset, dtype):
#         mean = dataset.mean()
#         std  = dataset.std()
#         return ((dataset - mean) / std).astype(dtype)

#     def __getitem__(self, index):
#         labelPath = self.labelPath[index]
#         filename = labelPath.split('/')[-1]
#         self.name = filename
#         # imagePath = self.root + '/test/' +'/images/' + filename[:-4] + '-MRA.mha' ##.mha data## 
#         imagePath = self.root + '/test/' +'/image/' + filename[:-4] + '.nii'
#         # ## for .mha data ##
#         # img = sitk.ReadImage(imagePath)
#         # mask = sitk.ReadImage(labelPath)

#         ## for .nii data ##
#         img = nib.load(imagePath)
#         mask = nib.load(labelPath)

#         # ## for .mha data ##
#         # img = sitk.GetArrayFromImage(img).astype(np.float32)
#         # Image = self.standardization_intensity_normalization(img, 'float32')
#         # img = torch.from_numpy(np.ascontiguousarray(Image)).unsqueeze(0)
        
#         # mask = sitk.GetArrayFromImage(mask).astype(np.float32) 
#         # label = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0)
#         # label = label / 255

#         ## for .nii data ##
#         img = img.get_data().astype(np.float32)
#         Image = self.standardization_intensity_normalization(img, 'float32')
#         img = torch.from_numpy(np.ascontiguousarray(Image)).unsqueeze(0)
#         mask = mask.get_data().astype(np.float32)
#         label = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0)
#         #label = label / 255 ##for .mha data

#         return img, label

    

#     def __len__(self):
#         '''
#         返回总的图像数量
#         :return:
#         '''
#         return len(self.labelPath)

#     def get_dataPath(self, root, isTraining):
#         '''
#         依次读取输入图片和label的文件路径，并放到array中返回
#         :param root: 存放的文件夹
#         :return:
#         '''
#         if isTraining:
#             root = os.path.join(root + "/train/label")
#             imgPath = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
#         else:
#             root = os.path.join(root + "/test/label")#/home/imed/disk5TA/xyx/pytorch/MRABrain/test/mesh_label
#             imgPath = list(map(lambda x: os.path.join(root, x), os.listdir(root)))

#         # print(len(imgPath))

#         return imgPath

#     def getFileName(self):
#         return self.name
import torch.utils.data as data
import numpy as np
import os
from PIL import Image
from torchvision import transforms
import SimpleITK as sitk
import torch
import nibabel as nib


class MRADataset(data.Dataset):

    def __init__(self, root, isTraining=True):
        self.labelPath = self.get_dataPath(root, False)
        self.imgPath = root
        self.isTraining = isTraining
        self.name = ' '
        self.root = root

    def standardization_intensity_normalization(self,dataset, dtype):
        mean = dataset.mean()
        std  = dataset.std()
        return ((dataset - mean) / std).astype(dtype)

    def __getitem__(self, index):
        labelPath = self.labelPath[index]
        filename = labelPath.split('/')[-1]
        self.name = filename
        imagePath = self.root + '/test/' +'/images/' + filename[:-4] + '-MRA.mha'## for .mha
        # imagePath = self.root + '/test/' +'/image/' + filename[:-4] + '.nii'
        # ## for .mha data ##
        img = sitk.ReadImage(imagePath)
        mask = sitk.ReadImage(labelPath)

        ## for .nii data ##
        # img = nib.load(imagePath)
        # mask = n