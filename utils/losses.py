import random
import numpy as np
from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F

# from monai.losses.tversky import TverskyLoss
from torch.autograd import Variable
from itertools import  filterfalse as ifilterfalse


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


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)                            # sigma = 1.5    shape: [11, 1]
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)    # unsqueeze()函数,增加维度  .t() 进行了转置 shape: [1, 1, 11, 11]
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()   # window shape: [1,1, 11, 11]
    return window


def std(img,  window_size=9):
    padd = window_size // 2
    channel = img.shape[1]
    window = create_window(window_size, channel=channel).to(img.device)
    mu = F.conv2d(img, window, padding=padd, groups=channel)
    mu_sq = mu.pow(2)
    sigma = F.conv2d(img * img, window, padding=padd, groups=channel) - mu_sq
    return sigma


class std_loss:
    def __init__(self):
        self.l1_loss = nn.L1Loss()

    def __call__(self, fuse, img1, img2):
        std1 = std(img1)
        std2 = std(img2)

        map = torch.where((std1 - std2) > 0, 1, 0)
        # w1 = std1 / (std1 + std2 + 1e-6)
        # w2 = std2 / (std1 + std2 + 1e-6)

        joint_img = map * img1 + (1 - map) * img2
        loss = self.l1_loss(fuse, joint_img)
        
        return loss


class fusion_loss:
    def __init__(self, weight):
        self.weight = weight
        self.l1_loss = nn.L1Loss()
        # self.patch_loss = patch_loss(64)
        self.std_loss = std_loss()
    
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

        # con_loss = self.patch_loss(Y_fuse, Y_img1, img2[:,0:1,:,:], Mask)
        con_loss = self.std_loss(Y_fuse, Y_img1, img2[:,0:1,:,:])
        gradient_loss = self.l1_loss(fuse_grad_x, joint_grad_x) + self.l1_loss(fuse_grad_y, joint_grad_y)
        color_loss = self.l1_loss(Cr_fuse, Cr_img1) + self.l1_loss(Cb_fuse, Cb_img1)
        
        loss = self.weight[0] * con_loss  + self.weight[1] * gradient_loss  + self.weight[2] * color_loss
        return loss


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n

 
class seg_loss:
    def __init__(self, weight, num_classes, ignore_index=255): 
        self.weight = weight
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=ignore_index)
        # self.tversky_loss = TverskyLoss(alpha=0.3, softmax=True, reduction='mean')
    
    def get_one_hot(self, inputs, ignore_index=255):
        mask = (inputs>=0).float() * (inputs!=ignore_index).float()
        inputs[inputs==ignore_index] = 0
        inputs = F.one_hot(inputs, self.num_classes).permute(0, 3, 1, 2)
        inputs = inputs * mask.unsqueeze(1)

        return inputs
    
    def lovasz_softmax_loss(self, prediction, gt, ignore_index=255):
        prediction = F.softmax(prediction, dim=1)
        loss = lovasz_softmax(prediction, gt, ignore=ignore_index)
        
        return loss

    def __call__(self, prediction, gt):
        _, H, W = gt.size()
        prediction = F.interpolate(prediction, size=(H, W), mode='bilinear', align_corners=False)
        loss1 = self.ce_loss(prediction, gt)
        loss2 = self.lovasz_softmax_loss(prediction, gt)

        loss = self.weight[0] * loss1 + self.weight[1] * loss2
        return loss

if __name__ == "__main__":
    gt = torch.tensor([[[1,3,255],[1,1,2],[1,1,2]],[[1,1,255],[1,1,2],[1,3,2]]])
    print(gt)
    ignore_index = 255
    mask = (gt>=0).float() * (gt!=ignore_index).float()
    print(mask)
    gt[gt==ignore_index] = 0
    gt = F.one_hot(gt).permute(0, 3, 1, 2)
    gt = gt * mask.unsqueeze(1)

    print(gt)