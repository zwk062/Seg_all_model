import SimpleITK as sitk 
import numpy as np
from skimage import filters
file="/home/imed/segmentation/MICCAI_code/03.mha"
image=sitk.ReadImage(file)
data=sitk.GetArrayFromImage(image) * 255
print(np.unique(data),data.shape)
ne = sum(data == 2)
print (ne, ne.shape)
# count = (mask_full_binarization.reshape(-1, 3)[:,1] == 255).sum()
# print (count)

threshold = filters.threshold_otsu(data, nbins=256)

pred = np.where(data > threshold, 255.0, 0)
pred = pred / 255
# sitk.WriteImage(pred, os.path.join(save_path + filename + ".mha"))
pred = sitk.GetImageFromArray(pred)
sitk.WriteImage(pred, "/home/imed/segmentation/MICCAI_code/033.mha")