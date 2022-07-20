import random
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
import os
# from turbojpeg import TurboJPEG
from PIL import Image
from jpeg2dct.numpy import load, loads
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import albumentations as A

class LQGTDataset(data.Dataset):
    '''
    Read LQ (Low Quality, here is LR) and GT image pairs.
    If only GT image is provided, generate LQ image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt, dataset_opt):
        super(LQGTDataset, self).__init__()
        self.transforms = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.1),
                # A.RandomCrop(height=256,width=256,p=1.0),
                # A.ColorJitter(),
                A.RandomBrightnessContrast(p=0.2),
                A.JpegCompression(quality_lower=70, quality_upper=100,p=0.5),
                A.RGBShift(r_shift_limit=25,g_shift_limit=25,b_shift_limit=25,p=0.5),
            ]
        )
        self.opt = opt
        self.dataset_opt = dataset_opt
        self.paths_LQ, self.paths_GT = None, None
        self.sizes_LQ, self.sizes_GT = None, None
        self.jpeg_filepath=['/home/qichaoying/Documents/COCOdataset/jpeg_train2017/train_2017_10',
                            '/home/qichaoying/Documents/COCOdataset/jpeg_train2017/train_2017_30',
                            '/home/qichaoying/Documents/COCOdataset/jpeg_train2017/train_2017_50',
                            '/home/qichaoying/Documents/COCOdataset/jpeg_train2017/train_2017_70',
                            '/home/qichaoying/Documents/COCOdataset/jpeg_train2017/train_2017_90']
        self.paths_GT, self.sizes_GT = util.get_image_paths(dataset_opt['dataroot_GT'])

        assert self.paths_GT, 'Error: GT path is empty.'

        self.random_scale_list = [1]

        # self.jpeg = TurboJPEG('/usr/lib/libturbojpeg.so')


    def __getitem__(self, index):

        scale = self.dataset_opt['scale']
        GT_size = self.dataset_opt['GT_size']

        # get GT image
        GT_path = self.paths_GT[index]

        # img_GT = util.read_img(GT_path)
        img_GT = cv2.imread(GT_path, cv2.IMREAD_COLOR)
        img_GT = util.channel_convert(img_GT.shape[2], self.dataset_opt['color'], [img_GT])[0]
        img_GT = self.transforms(image=img_GT)["image"]
        img_GT = img_GT.astype(np.float32) / 255.
        if img_GT.ndim == 2:
            img_GT = np.expand_dims(img_GT, axis=2)
        # some images have 4 channels
        if img_GT.shape[2] > 3:
            img_GT = img_GT[:, :, :3]


        # filepath, tempfilepath = os.path.split(GT_path)

        # load_jpeg = False
        # if load_jpeg:
        #     index = np.random.randint(0,5)
        #     # index = 0
        #     if index==0:
        #         QF=0.1
        #     elif index==1:
        #         QF=0.3
        #     elif index == 2:
        #         QF = 0.5
        #     elif index == 3:
        #         QF = 0.7
        #     else: #if index==4:
        #         QF= 0.9
        #     jpeg_path = os.path.join(self.jpeg_filepath[index],tempfilepath)
        #
        #     label = int((QF*10)/2)
        #     # label = torch.zeros(5)
        #     # label[int((QF*10)/2)] = 1
        #     # print(jpeg_path)
        #
        #     ######### if exist jpeg path
        #     img_jpeg_GT = util.read_img(jpeg_path)
        #     ######### else
        #     # img_jpeg_GT = util.read_img(GT_path)


        # img_jpeg_GT = util.channel_convert(img_jpeg_GT.shape[2], self.dataset_opt['color'], [img_jpeg_GT])[0]


        ###### directly resize instead of crop
        img_GT = cv2.resize(np.copy(img_GT), (GT_size, GT_size),
                            interpolation=cv2.INTER_LINEAR)
        # img_jpeg_GT = cv2.resize(np.copy(img_jpeg_GT), (GT_size, GT_size),
        #                          interpolation=cv2.INTER_LINEAR)


        orig_height, orig_width, _ = img_GT.shape
        H, W, _ = img_GT.shape

        img_gray = rgb2gray(img_GT)
        sigma = 2 #random.randint(1, 4)

        if self.opt['model']=="PAMI" or self.opt['model']=="CLRNet":
            canny_img = canny(img_gray, sigma=sigma, mask=None)
            canny_img = canny_img.astype(np.float)
            canny_img = self.to_tensor(canny_img)
        # elif self.opt['model']=="ICASSP_NOWAY":
        #     canny_img = img_gray
        else:
            canny_img = None


        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            # img_jpeg_GT = img_jpeg_GT[:, :, [2, 1, 0]]
            # img_LQ = img_LQ[:, :, [2, 1, 0]]


        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        # canny_img = torch.from_numpy(np.ascontiguousarray(canny_img)).float()
        # img_jpeg_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_jpeg_GT, (2, 0, 1)))).float()
        # img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()

        # if LQ_path is None:
        #     LQ_path = GT_path

        return (img_GT, 0, canny_img if canny_img is not None else img_GT.clone())

    def __len__(self):
        return len(self.paths_GT)

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t
