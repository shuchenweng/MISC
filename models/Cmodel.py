import torch
import torch.nn as nn
import torch.nn.functional as F
from miscc.config import cfg
from models.Blocks import conv1x1, encode_image_by_ntimes

class COLORG_NET_PIXEL(nn.Module):
    def __init__(self):
        super(COLORG_NET_PIXEL, self).__init__()
        ndf = cfg.GAN.DF_DIM
        self.height = cfg.TREE.BASE_SIZE_HEIGHT * 4
        self.width = cfg.TREE.BASE_SIZE_WIDTH * 4
        self.img_code = encode_image_by_ntimes(ngf=0, ndf=ndf, n_layer=4, 
            kernel_size=3, stride=1, padding=1, up_pow=1, use_spn=False)
        self.conv = conv1x1(ndf,6)

    def forward(self, x):
        x_code16 = self.img_code(x)
        paras = self.conv(x_code16)
        x = x * paras[:,:3,:,:] + paras[:,3:,:,:]
        x = nn.Tanh()(x)
        return x