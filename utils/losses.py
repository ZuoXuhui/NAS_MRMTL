import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli


def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat([Y, Cr, Cb], dim=1).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


def Sobelxy(x):
    kernelx = [[-1, 0, 1],
              [-2,0 , 2],
              [-1, 0, 1]]
    kernely = [[1, 2, 1],
              [0,0 , 0],
              [-1, -2, -1]]
    kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
    kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
    weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
    weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    
    sobelx=F.conv2d(x, weightx, padding=1)
    sobely=F.conv2d(x, weighty, padding=1)
    #return torch.abs(sobelx)+torch.abs(sobely)
    return sobelx, sobely


def combine_sobel_xy(img):
    R_img = img[:,0:1,:,:]
    B_img = img[:,1:2,:,:]
    G_img = img[:,2:,:,:]
    R_img_grad_x, R_img_grad_y = Sobelxy(R_img)
    B_img_grad_x, B_img_grad_y = Sobelxy(B_img)
    G_img_grad_x, G_img_grad_y = Sobelxy(G_img)
    
    grad_x = torch.cat([R_img_grad_x, B_img_grad_x, G_img_grad_x], 1)
    grad_y = torch.cat([R_img_grad_y, B_img_grad_y, G_img_grad_y], 1)
    return grad_x, grad_y


def get_random_patch_coordinates(mask: torch.Tensor, patch_height: int, patch_width: int):
    B, _, H, W = mask.shape
    coordinates = []

    for b in range(B):
        single_channel_mask = mask[b].max(dim=0).values
        mask_indices = torch.nonzero(single_channel_mask, as_tuple=False)

        top_left_h = torch.min(mask_indices[:, 0]).item()
        top_left_w = torch.min(mask_indices[:, 1]).item()
        bottom_right_h = torch.max(mask_indices[:, 0]).item()
        bottom_right_w = torch.max(mask_indices[:, 1]).item()

        max_h = min(bottom_right_h - patch_height + 1, H - patch_height)
        max_w = min(bottom_right_w - patch_width + 1, W - patch_width)
        min_h = max(top_left_h, 0)
        min_w = max(top_left_w, 0)

        available_positions = []
        for h in range(min_h, max_h + 1, patch_height):
            for w in range(min_w, max_w + 1, patch_width):
                if h + patch_height <= H and w + patch_width <= W:
                    patch_mask_area = single_channel_mask[h:h + patch_height, w:w + patch_width]
                    if torch.all(patch_mask_area > 0):
                        available_positions.append((h, w))
        
        random.shuffle(available_positions)
        coordinates.append(available_positions)
    
    return coordinates


def extract_patches_from_image(image: torch.Tensor, all_coordinates: torch.Tensor, patch_height: int, patch_width: int):
    B, C, H, W = image.shape
    
    N = min(len(coordinates) for coordinates in all_coordinates)
    P = patch_height * patch_width
    
    all_patches = []
    for b in range(B):
        b_image = image[b]
        coordinates = all_coordinates[b]

        patches = torch.zeros((N, C, P), dtype=image.dtype, device=image.device)

        for i in range(N):
            (h, w) = coordinates[i]
            patch = b_image[:, h:h + patch_height, w:w + patch_width]
            patches[i] = patch.reshape(C, -1)

        all_patches.append(patches)

    batch_patches = torch.stack(all_patches, dim=0).to(image.device, image.dtype)

    return batch_patches


class patch_loss(nn.Module):
    def __init__(self, patch_size=16):
        super().__init__()
        # patch_size = 16 for image size 256*256
        self.patch_size = patch_size
        self.l1_loss = nn.L1Loss()
    
    def forward(self, fuse, img1, img2, Mask):
        # 1 - ir, 2 - vi
        B, C, H, W = fuse.shape

        # patch_matrix  # (b, c, N, patch_size * patch_size)
        coordinates = get_random_patch_coordinates(Mask, self.patch_size, self.patch_size)
        patch_fuse = extract_patches_from_image(fuse, coordinates, self.patch_size, self.patch_size)
        patch_img1 = extract_patches_from_image(img1, coordinates, self.patch_size, self.patch_size)
        patch_img2 = extract_patches_from_image(img2, coordinates, self.patch_size, self.patch_size)

        b0, c0, n0, p0 = patch_fuse.shape
        b1, c1, n1, p1 = patch_img1.shape
        b2, c2, n2, p2 = patch_img2.shape
        assert n0 == n1 == n2 and p0 == p1 == p2, \
                f"The number of patches ({n0}, {n1} and {n2}) or the patch sice ({p0}, {p1} and {p2}) doesn't match ."

        mu1 = torch.mean(patch_img1, dim=3)
        mu2 = torch.mean(patch_img2, dim=3)

        mu1_re = mu1.view(b1, c1, n1, 1).repeat(1, 1, 1, p1)
        mu2_re = mu2.view(b2, c2, n2, 1).repeat(1, 1, 1, p2)

        # SD, b1 * c1 * n1 * 1
        sd1 = torch.sqrt(torch.sum(((patch_img1 - mu1_re) ** 2), dim=3) / p1)
        sd2 = torch.sqrt(torch.sum(((patch_img2 - mu2_re) ** 2), dim=3) / p2)
        # sd_mask = getBinaryTensor(sd1 - sd2, 0)

        w1 = sd1 / (sd1 + sd2 + 1e-6)
        w2 = sd2 / (sd1 + sd2 + 1e-6)
        
        w1 = w1.view(b1, c1, n1, 1).repeat(1, 1, 1, p1)
        w2 = w2.view(b2, c2, n2, 1).repeat(1, 1, 1, p2)

        loss = self.l1_loss(patch_fuse, w1 * patch_img1 + w2 * patch_img2)
        return loss

class fusion_loss:
    def __init__(self, weight):
        self.weight = weight
        self.l1_loss = nn.L1Loss()
        self.patch_loss = patch_loss(32)
    
    def __call__(self, fuse, img1, img2, Mask):
        ## img1 is RGB, img2 is Gray
        fuse = fuse * Mask

        YCbCr_fuse = RGB2YCrCb(fuse)
        Y_fuse = YCbCr_fuse[:,0:1,:,:]
        Cr_fuse = YCbCr_fuse[:,1:2,:,:]
        Cb_fuse = YCbCr_fuse[:,2:,:,:]

        YCbCr_img1 = RGB2YCrCb(img1)
        Y_img1 = YCbCr_img1[:,0:1,:,:]
        Cr_img1 = YCbCr_img1[:,1:2,:,:]
        Cb_img1 = YCbCr_img1[:,2:,:,:]

        fuse_grad_x, fuse_grad_y = Sobelxy(Y_fuse)
        img1_grad_x, img1_grad_y = Sobelxy(Y_img1)
        img2_grad_x, img2_grad_y = Sobelxy(img2[:,0:1,:,:])

        joint_grad_x = torch.maximum(img1_grad_x, img2_grad_x)
        joint_grad_y = torch.maximum(img1_grad_y, img2_grad_y)

        con_loss = self.patch_loss(Y_fuse, Y_img1, img2[:,0:1,:,:], Mask)
        gradient_loss = self.l1_loss(fuse_grad_x, joint_grad_x) + self.l1_loss(fuse_grad_y, joint_grad_y)
        color_loss = self.l1_loss(Cr_fuse, Cr_img1) + self.l1_loss(Cb_fuse, Cb_img1)

        loss = self.weight[0] * con_loss  + self.weight[1] * gradient_loss  + self.weight[2] * color_loss
        return loss

class seg_loss:
    def __init__(self, ignore_index=255):    
        self.semantic_loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=ignore_index)

    def __call__(self, prediction, gt):
        _, H, W = gt.size()
        prediction = F.interpolate(prediction, size=(H, W), mode='bilinear', align_corners=False)
        loss = self.semantic_loss(prediction, gt)

        return loss
