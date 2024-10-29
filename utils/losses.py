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


class fusion_loss:
    def __init__(self, weight):
        self.weight = weight
        self.l1_loss = nn.L1Loss()
    
    def __call__(self, fuse, img1, img2, Mask):
        ## img1 is RGB, img2 is Gray
        fuse = fuse * Mask

        YCbCr_fuse = RGB2YCrCb(fuse)
        Cr_fuse = YCbCr_fuse[:,1:2,:,:]
        Cb_fuse = YCbCr_fuse[:,2:,:,:]

        YCbCr_img1 = RGB2YCrCb(img1)
        Cr_img1 = YCbCr_img1[:,1:2,:,:]
        Cb_img1 = YCbCr_img1[:,2:,:,:]

        fuse_grad_x, fuse_grad_y = combine_sobel_xy(fuse)
        img1_grad_x, img1_grad_y = combine_sobel_xy(img1)
        img2_grad_x, img2_grad_y = combine_sobel_xy(img2)

        joint_grad_x = torch.maximum(img1_grad_x, img2_grad_x)
        joint_grad_y = torch.maximum(img1_grad_y, img2_grad_y)
        joint_int  = torch.maximum(img1, img2)

        con_loss = self.l1_loss(fuse, joint_int)
        gradient_loss = 0.5*self.l1_loss(fuse_grad_x, joint_grad_x) + 0.5*self.l1_loss(fuse_grad_y, joint_grad_y)
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
