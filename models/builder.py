import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tasks import SegTask, FusionTask

from .encoders.dual_segformer import RGBXTransformerBlock
from .common_layers import get_nddr
from .decoders.MLPDecoder import DecoderHead
from .decoders.CNNDecoder import CNNHead

from utils import AttrDict


class SingleTaskNet(nn.Module):
    def __init__(self, cfg, norm_layer=nn.BatchNorm2d):
        super(SingleTaskNet, self).__init__()
        ## task auxiliary training
        self.task1 = SegTask(cfg.num_classes, cfg.datasets.ignore_index)
        self.task2 = FusionTask(cfg.weights.task2)

        ## params
        self.cfg = cfg
        ## define network
        self.num_classes =  cfg.num_classes
        self.in_chans = cfg.in_chans
        # encoder
        self.num_stages = cfg.num_stages
        img_size = cfg.encoder.img_size
        patch_size = cfg.encoder.patch_size
        stride = cfg.encoder.stride
        depths = cfg.encoder.depths
        embed_dims = cfg.encoder.embed_dims
        num_heads = cfg.encoder.num_heads
        mlp_ratios = cfg.encoder.mlp_ratios
        sr_ratios = cfg.encoder.sr_ratios
        
        dpr = [x.item() for x in torch.linspace(0, cfg.encoder.drop_path_rate, sum(depths))]

        cur = 0
        encoders = []
        for stage_id in range(self.num_stages):
            encoder = RGBXTransformerBlock(img_size=img_size, patch_size=patch_size[stage_id], stride=stride[stage_id], depths=depths[stage_id],
                                           in_chans=self.in_chans if stage_id == 0 else embed_dims[stage_id-1], embed_dims=embed_dims[stage_id], 
                                           num_heads=num_heads[stage_id], mlp_ratios=mlp_ratios[stage_id], qkv_bias=cfg.encoder.qkv_bias,
                                           qk_scale=cfg.encoder.qk_scale, drop_rate=cfg.encoder.drop_rate, attn_drop_rate=cfg.encoder.attn_drop_rate,
                                           drop_path_rate=dpr[cur:cur+depths[stage_id]], sr_ratios=sr_ratios[stage_id])
            cur += depths[stage_id]
            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)
        # decoder
        self.task1_head = DecoderHead(in_channels=cfg.decoder.channels, num_classes=cfg.num_classes, norm_layer=norm_layer, embed_dim=cfg.decoder.embed_dim)
        self.task2_head = CNNHead(in_channels=cfg.decoder.channels)
        
        self._step = 0
        # load from
        self.init_weigths(cfg.get('load_from', None))

    def step(self):
        self._step += 1

    def init_weigths(self, model_file):
        if model_file is None:
            return
        if isinstance(model_file, str):
            state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        else:
            state_dict = model_file

        self.load_state_dict(state_dict, strict=False)
        del state_dict

    def loss(self, modal_x, modal_y, label, Mask):
        result = self.forward(modal_x, modal_y)
        result.loss1 = self.task1.loss(result.out1, label)
        result.loss2 = self.task2.loss(result.out2, modal_x, modal_y, Mask)
        result.loss = result.loss1 + self.cfg.weights.TASK2_FACTOR * result.loss2
        return result

    def forward(self, modal_x, modal_y):
        original_input = [modal_x, modal_y]
        out_semantic = []
        out_visual = []
        for stage_id in range(self.num_stages):
            modal_x, modal_y = self.encoders[stage_id].forward(modal_x, modal_y)
            out_visual.append(modal_x)
            out_visual.append(modal_y)
            fuse_xy = self.encoders[stage_id].forward_features(modal_x, modal_y)
            out_semantic.append(fuse_xy)

        out1 = self.task1_head.forward(out_semantic)
        out2 = self.task2_head.forward(out_visual, original_input)

        return AttrDict({'out1': out1, 'out2': out2})

class NDDRTaskNet(nn.Module):
    def __init__(self, cfg, norm_layer=nn.BatchNorm2d):
        super(NDDRTaskNet, self).__init__()
        ## task auxiliary training
        self.task1 = SegTask(cfg.num_classes, cfg.datasets.ignore_index)
        self.task2 = FusionTask(cfg.weights.task2)

        ## params
        self.cfg = cfg
        ## define network
        self.num_classes =  cfg.num_classes
        self.in_chans = cfg.in_chans
        # encoder
        self.num_stages = cfg.num_stages
        img_size = cfg.encoder.img_size
        patch_size = cfg.encoder.patch_size
        stride = cfg.encoder.stride
        depths = cfg.encoder.depths
        embed_dims = cfg.encoder.embed_dims
        num_heads = cfg.encoder.num_heads
        mlp_ratios = cfg.encoder.mlp_ratios
        sr_ratios = cfg.encoder.sr_ratios
        
        dpr = [x.item() for x in torch.linspace(0, cfg.encoder.drop_path_rate, sum(depths))]

        cur = 0
        encoders = []
        for stage_id in range(self.num_stages):
            encoder = RGBXTransformerBlock(img_size=img_size, patch_size=patch_size[stage_id], stride=stride[stage_id], depths=depths[stage_id],
                                           in_chans=self.in_chans if stage_id == 0 else embed_dims[stage_id-1], embed_dims=embed_dims[stage_id], 
                                           num_heads=num_heads[stage_id], mlp_ratios=mlp_ratios[stage_id], qkv_bias=cfg.encoder.qkv_bias,
                                           qk_scale=cfg.encoder.qk_scale, drop_rate=cfg.encoder.drop_rate, attn_drop_rate=cfg.encoder.attn_drop_rate,
                                           drop_path_rate=dpr[cur:cur+depths[stage_id]], sr_ratios=sr_ratios[stage_id])
            cur += depths[stage_id]
            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)
        
        # nddr
        nddrs = []
        for stage_id in range(self.num_stages):
            out_channels = embed_dims[stage_id]
            nddr = get_nddr(cfg, out_channels, out_channels)
            nddrs.append(nddr)
        self.nddrs = nn.ModuleList(nddrs)

        # decoder
        self.task1_head = DecoderHead(in_channels=cfg.decoder.channels, num_classes=cfg.num_classes, norm_layer=norm_layer, embed_dim=cfg.decoder.embed_dim)
        self.task2_head = CNNHead(in_channels=cfg.decoder.channels)
        
        self._step = 0

    def step(self):
        self._step += 1

    def loss(self):
        pass

    def forward(self, img1, img2):
        pass