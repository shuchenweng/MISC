import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from miscc.config import cfg
from models.Amodel import GlobalAttentionGeneral

from models.Blocks import GLU, downBlock_G, upBlock, conv3x3, PixelNorm, ResBlock_ADIN
from models.Cmodel import COLORG_NET_PIXEL

class INIT_STAGE_G_PAT(nn.Module):
    def __init__(self, ngf):
        super(INIT_STAGE_G_PAT, self).__init__()
        self.gf_dim = ngf
        self.in_dim = cfg.GAN.Z_DIM  # cfg.TEXT.EMBEDDING_DIM
        self.height = cfg.TREE.BASE_SIZE_HEIGHT//8
        self.width = cfg.TREE.BASE_SIZE_WIDTH//8

        self.define_module()

    def define_module(self):
        nz, ngf = self.in_dim, self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(nz, ngf * self.height * self.width * 2, bias=False),
            nn.BatchNorm1d(ngf * self.height * self.width * 2),
            GLU())

        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)

    def forward(self, z_code):
        """
        :param z_code: batch x cfg.GAN.Z_DIM
        :param c_code: batch x cfg.TEXT.EMBEDDING_DIM
        :return: batch x ngf/4 x 32 x 32
        """
        # state size ngf x 4 x 4
        out_code = self.fc(z_code)
        out_code = out_code.view(-1, self.gf_dim, self.height, self.width)
        # state size ngf/2 x 16 x 16
        out_code = self.upsample1(out_code)
        # state size ngf/4 x 32 x 32
        out_code = self.upsample2(out_code)

        return out_code

class INIT_STAGE_G_SEG(nn.Module):
    def __init__(self, ngf, ncf):
        super(INIT_STAGE_G_SEG, self).__init__()
        self.gf_dim = ngf # the base number of channels: e.g., 12
        self.in_dim = ncf # number of object parts of interest
        self.define_module()

    def define_module(self):
        ncf, ngf = self.in_dim, self.gf_dim
        # Convolution-InstanceNorm-ReLU
        self.conv3x3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ncf, ngf, kernel_size=3, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True))

        self.downsample1 = downBlock_G(ngf, ngf*2) # output 24 channels
        self.downsample2 = downBlock_G(ngf*2, ngf*4) # output 48 channels

    def forward(self, seg):
        """
        :param seg: batch x ncf x seg_size (128) x seg_size
        :return: batch x ngf*4 x 64 x 64
        """
        # state size ngf x 128 x 128
        out_code = self.conv3x3(seg)
        # state size ngf*2 x 64 x 64
        out_code = self.downsample1(out_code)
        # state size ngf*4 x 32 x 32
        out_code = self.downsample2(out_code)

        return out_code

class INIT_STAGE_G_MAIN(nn.Module):
    def __init__(self, ngf):
        super(INIT_STAGE_G_MAIN, self).__init__()
        self.gf_dim = ngf # the base number of channels: e.g., 12

        self.define_module()

    def define_module(self):
        ngf = self.gf_dim
        init_channel = ngf * 2
        self.layer_1 = ResBlock_ADIN(init_channel)
        self.layer_2 = ResBlock_ADIN(init_channel)
        self.upsample = upBlock(init_channel, ngf)

    def forward(self, z_code, h_code_seg, h_code_att, h_code_pat, seg=None):
        """
        :param h_code_seg: batch x ngf x 32 x 32
        :param h_code_pat: batch x ngf x 32 x 32
        :return: batch x ngf*4 x 64 x 64
        """
        # state size ngf*2 x 32 x 32
        out_code = torch.cat((h_code_seg, h_code_att), dim=1)
        out_code = self.layer_1(out_code, z_code, seg)
        out_code = self.layer_2(out_code, z_code, seg)
        # state size ngf x 64 x 64
        out_code = self.upsample(out_code)

        return out_code


class NEXT_STAGE_G(nn.Module):
    def __init__(self, ngf, nef):
        super(NEXT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.ef_dim = nef
        self.define_module()

    def define_module(self):
        ngf = self.gf_dim
        self.attnet = GlobalAttentionGeneral(ngf, self.ef_dim)
        init_channel = ngf * 2
        self.layer_1 = ResBlock_ADIN(init_channel)
        self.layer_2 = ResBlock_ADIN(init_channel)

        self.upsample = upBlock(init_channel, ngf)

    def forward(self, z_code, h_code, att_embs, seg, pooled_seg):
        """
            h_code1(query):  batch x idf x ih x iw (queryL=ihxiw)
            att_embs(context): batch x cdf x sourceL (sourceL=seq_len)
            seg: batch x pncf x ih x iw
            pooled_seg: batch x ih x iw
            attn: batch x sourceL x queryL
        """
        c_code, attn = self.attnet(h_code, att_embs, seg, pooled_seg) ##########
        h_c_code = torch.cat((h_code, c_code), 1)
        out_code = self.layer_1(h_c_code, z_code, seg)
        out_code = self.layer_2(out_code, z_code, seg)

        # state size ngf/2 x 2in_size x 2in_size
        out_code = self.upsample(out_code)

        return out_code, attn

class Z_MLP(nn.Module):
    def __init__(self, in_dim, n_mlp):
        super(Z_MLP, self).__init__()
        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(nn.Linear(in_dim, in_dim))
            layers.append(nn.LeakyReLU(0.2))

        self.style = nn.Sequential(*layers)
    def forward(self, x):
        return self.style(x)

class GET_IMAGE_G(nn.Module):
    def __init__(self, ngf):
        super(GET_IMAGE_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, 3),
            nn.Tanh()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


class G_NET(nn.Module):
    def __init__(self):
        super(G_NET, self).__init__()
        ngf = cfg.GAN.GF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        pncf = cfg.GAN.P_NUM
        if cfg.TREE.BRANCH_NUM > 0:
            self.z_net1_mlp = Z_MLP(cfg.GAN.Z_DIM, cfg.GAN.Z_MLP_NUM)
            self.h_net1_seg = INIT_STAGE_G_SEG(ngf//4, pncf)
            self.h_net1_main = INIT_STAGE_G_MAIN(ngf)
            self.h_net_att = GlobalAttentionGeneral(ngf, nef)
            self.img_net1 = GET_IMAGE_G(ngf)

        # gf x 64 x 64
        if cfg.TREE.BRANCH_NUM > 1:
            self.h_net2 = NEXT_STAGE_G(ngf, nef)
            self.img_net2 = GET_IMAGE_G(ngf)
        if cfg.TREE.BRANCH_NUM > 2:
            self.h_net3 = NEXT_STAGE_G(ngf, nef)
            self.img_net3 = GET_IMAGE_G(ngf)
            self.colorG = COLORG_NET_PIXEL()

    def forward(self, z_code, att_embs, segs, pooled_segs, imgs):
        """
            :param z_code: batch x cfg.GAN.Z_DIM
            :param glb_att_embs: batch x cfg.TEXT.EMBEDDING_DIM
            :param att_embs: batch x cdf x seq_len
            :param mask: batch x seq_len
            :param segs: [batch x pncf x seg_size x seg_size]
            :param pooled_segs: [batch x seg_size x seg_size]
            :return:
        """
        raw_fake_imgs = []
        attn_maps = []
        if cfg.TREE.BRANCH_NUM > 0:
            # seg feat
            h_net1_seg = self.h_net1_seg(segs[1])

            # att feat
            height_half, width_half = cfg.TREE.BASE_SIZE_HEIGHT//2, cfg.TREE.BASE_SIZE_WIDTH//2
            seg_half = nn.functional.interpolate(segs[0], (height_half, width_half), mode='nearest')
            pooled_seg_half = torch.max(seg_half, dim=1)[0]
            h_code_att, _ = self.h_net_att(h_net1_seg, att_embs, seg_half, pooled_seg_half)
            # pat feat
            h_code1_pat = None
            z_code = self.z_net1_mlp(z_code)
            # aggregation
            h_code1 = self.h_net1_main(z_code, h_net1_seg, h_code_att, h_code1_pat, seg_half)
            raw_fake_img1 = self.img_net1(h_code1)
            raw_fake_imgs.append(raw_fake_img1)
        if cfg.TREE.BRANCH_NUM > 1:
            h_code2, attn1 = self.h_net2(z_code, h_code1, att_embs, segs[0], pooled_segs[0])
            raw_fake_img2 = self.img_net2(h_code2)
            raw_fake_imgs.append(raw_fake_img2)
            if attn1 is not None:
                attn_maps.append(attn1)
        if cfg.TREE.BRANCH_NUM > 2:
            h_code3, attn2 = self.h_net3(z_code, h_code2, att_embs, segs[1], pooled_segs[1])
            raw_fake_img3 = self.img_net3(h_code3)
            raw_fake_imgs.append(raw_fake_img3)
            # background
            back_index = (pooled_segs[2] == 0).unsqueeze(dim=1).expand_as(imgs[2])
            front_index = (pooled_segs[2] == 1).unsqueeze(dim=1).expand_as(imgs[2])
            raw_fake_img_background = torch.zeros(raw_fake_img3.shape).cuda()
            raw_fake_img_background[back_index] = imgs[2][back_index]
            raw_fake_img_background[front_index] = raw_fake_img3[front_index]  # [batch_size, channel, height, width]
            fake_img = self.colorG(raw_fake_img_background)
            if attn2 is not None:
                attn_maps.append(attn2)

        return raw_fake_imgs, fake_img, attn_maps