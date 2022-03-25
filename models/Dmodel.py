import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from miscc.config import cfg
from miscc.manu_data import split_max_num
from models.Blocks import Block3x3_leakRelu, encode_image_by_ntimes, DownSample

class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf, nef, bcondition=False):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.bcondition = bcondition
        if self.bcondition:
            self.jointConv = Block3x3_leakRelu(ndf * 8 + nef, ndf * 8)

        self.outlogits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2),
            nn.Sigmoid())

    def forward(self, h_code, c_code=None):
        if self.bcondition and c_code is not None:
            # state size (ngf+egf) x 8 x 8
            h_c_code = torch.cat((h_code, c_code), 1)
            # state size ngf x in_size x in_size
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = h_code
        output = self.outlogits(h_c_code)
        return output

# For 64 x 64 images
class PAT_D_NET64(nn.Module):
    def __init__(self, b_jcu=True):
        super(PAT_D_NET64, self).__init__()
        ndf = cfg.GAN.DF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        self.img_code = encode_image_by_ntimes(ngf=0, ndf=ndf, n_layer=4, 
            kernel_size=4, stride=2, padding=1, use_spn=False)
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(ndf, nef*split_max_num, bcondition=True)

    def forward(self, x_var):
        x_code4 = self.img_code(x_var)  # 4 x 4 x 8df (cfg.GAN.LAYER_D_NUM=4)
        return x_code4


# For 128 x 128 images
class PAT_D_NET128(nn.Module):
    def __init__(self, b_jcu=True):
        super(PAT_D_NET128, self).__init__()
        ndf = cfg.GAN.DF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        self.img_code = encode_image_by_ntimes(ngf=0, ndf=ndf, n_layer=4, 
            kernel_size=4, stride=2, padding=1, use_spn=False)
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(ndf, nef*split_max_num, bcondition=True)

    def forward(self, x_var):
        x_code8 = self.img_code(x_var)  # 8 x 8 x 8df (cfg.GAN.LAYER_D_NUM=4)
        return x_code8


# For 256 x 256 images
class PAT_D_NET256(nn.Module):
    def __init__(self, b_jcu=True):
        super(PAT_D_NET256, self).__init__()
        ndf = cfg.GAN.DF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        self.img_code = encode_image_by_ntimes(ngf=0, ndf=ndf, n_layer=4, 
            kernel_size=4, stride=2, padding=1, use_spn=False)
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(ndf, nef*split_max_num, bcondition=True)

    def forward(self, x_var):
        x_code16 = self.img_code(x_var)  # 16 x 16 x 8df (cfg.GAN.LAYER_D_NUM=4)
        return x_code16

class SEG_D_NET(nn.Module):
    def __init__(self):
        super(SEG_D_NET, self).__init__()
        ndf = cfg.GAN.DF_DIM
        self.img_code = encode_image_by_ntimes(ngf=0, ndf=ndf, n_layer=2, 
            kernel_size=4, stride=2, padding=1, use_spn=True)
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf * 2, 1, 3, padding=1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.img_code(x)
        x = self.conv(x)
        return x