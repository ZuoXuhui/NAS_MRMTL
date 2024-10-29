import os
import numpy as np

import torch
from torch.utils.data import Dataset

import cv2
import mmcv

from .utils import normalize
from .utils import random_mirror, random_scale, generate_random_crop_pos, random_crop_pad_to_shape

class FusionDataset(Dataset):
    def __init__(
            self,
            data_root,
            path1="visible",
            path2="infrared",
            ):
        self.path1 = os.path.join(data_root, path1)
        self.path2 = os.path.join(data_root, path2)
        self.img_infos = self.load_annotations()

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)
    
    def load_annotations(self):
        img_infos = []
        for img in os.listdir(self.path1):
            img_name = img.split(".")[0]
            img_info = dict(filename=img_name)
            img_info['path1'] = os.path.join(self.path1, img)
            img_info['path2'] = os.path.join(self.path2, img)
            img_infos.append(img_info)
        
        return img_infos
    
    @staticmethod
    def _open_image(filepath, dtype=None):
        img = cv2.imread(filepath, -1)
        if len(img.shape) == 2:
            img = cv2.merge([img, img, img])
        elif len(img.shape) == 3:
            img = img[:,:,[2,1,0]]
        else:
            raise "Not Implement error"
        return np.array(img, dtype=dtype)
    
    @staticmethod
    def process_imgs(img):
        img = normalize(img)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img[None, :, :, :], dtype=np.float32)
        img = torch.FloatTensor(img)
        return img
    
    def __getitem__(self, idx):
        image_info = self.img_infos[idx]
        img1 = self._open_image(image_info["path1"])
        img2 = self._open_image(image_info["path2"])

        img1 = self.process_imgs(img1)
        img2 = self.process_imgs(img2)
        return img1, img2, image_info["filename"]


class CustomDataset(Dataset):
    def __init__(
            self,
            data_root,
            path1="visible",
            path1_suffix="png",
            path2="infrared",
            path2_suffix="png",
            label="labels",
            label_suffix="png",
            stage='train',
            split=None,
            ignore_index=None,
            preprocess=None,
            classes=None,
            palette=None
            ):
        self.stage = stage
        
        self.path1 = os.path.join(data_root, stage, path1)
        self.path2 = os.path.join(data_root, stage, path2)
        self.label = os.path.join(data_root, stage, label)

        self.img_infos = self.load_annotations(path1_suffix, path2_suffix, label_suffix, split)
        self.preprocess = preprocess

        self.ignore_index = ignore_index
        self.classes = classes
        self.palette = palette

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)
    
    def load_annotations(self, path1_suffix, path2_suffix, label_suffix, split=None):
        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name)
                    img_info['path1'] = os.path.join(self.path1, img_name+path1_suffix)
                    img_info['path2'] = os.path.join(self.path2, img_name+path2_suffix)
                    img_info['label'] = os.path.join(self.label, img_name+label_suffix)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(self.path1, path1_suffix, recursive=True):
                img_name = img.replace(path1_suffix, "")
                img_info = dict(filename=img_name)
                img_info['path1'] = os.path.join(self.path1, img_name+path1_suffix)
                img_info['path2'] = os.path.join(self.path2, img_name+path2_suffix)
                img_info['label'] = os.path.join(self.label, img_name+label_suffix)
                img_infos.append(img_info)

        return img_infos
    
    @staticmethod
    def _open_image(filepath, dtype=None):
        img = cv2.imread(filepath, -1)
        if len(img.shape) == 2:
            img = cv2.merge([img, img, img])
        elif len(img.shape) == 3:
            img = img[:,:,[2,1,0]]
        else:
            raise "Not Implement error"
        return np.array(img, dtype=dtype)
    
    @staticmethod
    def _open_label(filepath, dtype=np.uint8):
        img = cv2.imread(filepath, -1)
        return np.array(img, dtype=dtype)

    def __getitem__(self, idx):
        image_info = self.img_infos[idx]
        img1 = self._open_image(image_info["path1"], dtype=np.uint8)
        img2 = self._open_image(image_info["path2"], dtype=np.uint8)
        gt = self._open_label(image_info["label"], dtype=np.uint8)

        if self.preprocess is not None and self.stage == 'train':
            img1, img2, gt, Mask = self.preprocess(img1, img2, gt)
            img1 = torch.from_numpy(np.ascontiguousarray(img1)).float()
            img2 = torch.from_numpy(np.ascontiguousarray(img2)).float()
            gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
            
            output_dict = dict(modal_x=img1, modal_y=img2, label=gt, Mask=Mask, fn=image_info["filename"])

        else:
            output_dict = dict(modal_x=img1, modal_y=img2, label=gt, fn=image_info["filename"])
        
        return output_dict
        

class Train_pipline(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.train_scale_array = cfg.train.train_scale_array
        self.image_height = cfg.datasets.image_height
        self.image_width = cfg.datasets.image_width
    
    def __call__(self, img1, img2, gt):
        img1, img2, gt = random_mirror(img1, img2, gt)
        if self.train_scale_array is not None:
            img1, img2, gt, scale = random_scale(img1, img2, gt, self.train_scale_array)
        
        img1 = normalize(img1)
        img2 = normalize(img2)

        crop_size = (self.image_height, self.image_width)
        crop_pos = generate_random_crop_pos(img1.shape[:2], crop_size)

        p_img1, Margin = random_crop_pad_to_shape(img1, crop_pos, crop_size, 0)
        p_img2, _ = random_crop_pad_to_shape(img2, crop_pos, crop_size, 0)
        p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)

        p_img1 = p_img1.transpose(2, 0, 1)
        p_img2 = p_img2.transpose(2, 0, 1)

        Mask = np.zeros(p_img1.shape)
        Mask[:, Margin[0]:(crop_size[0]-Margin[1]), Margin[2]:(crop_size[1]-Margin[3])] = 1.

        return p_img1, p_img2, p_gt, Mask.astype(np.float32)