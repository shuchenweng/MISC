"""
Global attention takes a matrix and a query metrix.
Based on each query vector q, it computes a parameterized convex combination of the matrix
based.
H_1 H_2 H_3 ... H_n
  q   q   q       q
    |  |   |       |
      \ |   |      /
              .....
          \   |  /
                  a
Constructs a unit mapping.
$$(H_1 + H_n, q) => (a)$$
Where H is of `batch x n x dim` and q is of `batch x dim`.

References:
https://github.com/OpenNMT/OpenNMT-py/tree/fc23dfef1ba2f258858b2765d24565266526dc76/onmt/modules
http://www.aclweb.org/anthology/D15-1166
"""

import torch
import torch.nn as nn
import numpy as np
from miscc.utils import l1norm
from miscc.config import cfg
from miscc.manu_data import part2attr_np
from models.Blocks import conv1x1, conv1x1_1d

def cos_attention(A,B):
    batch_size = A.shape[0]
    idf = A.shape[1]
    A_dim = A.shape[2]
    B_dim = B.shape[2]
    A = A.permute((0,2,1)).reshape(batch_size*A_dim,idf)
    A = A / (A.norm(p=2, dim=1).reshape(-1, 1).expand_as(A)+1e-8)
    A = A.reshape(batch_size,A_dim,idf)
    B = B.permute((0,2,1)).reshape(batch_size*B_dim,idf)
    B = B / (B.norm(p=2, dim=1).reshape(-1, 1).expand_as(B)+1e-8)
    B = B.reshape(batch_size,B_dim,idf).permute((0,2,1))
    attn = torch.bmm(A,B) + 1
    return attn

class GlobalAttentionGeneral(nn.Module):
    def __init__(self, idf, cdf):
        super(GlobalAttentionGeneral, self).__init__()
        self.conv_context = conv1x1(cdf, idf)
        self.phi_key = conv1x1_1d(idf, idf)
        self.phi_value = conv1x1_1d(idf, idf)
        self.out_conv = conv1x1_1d(idf * 2, idf)

    def forward(self, input, context, seg, pooled_seg):
        """
            input: batch x idf x ih x iw (queryL=ihxiw)
            context: batch x cdf x sourceL
        """
        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context.size(0), context.size(2)
        # batch x cdf x sourceL --> batch x cdf x sourceL x 1
        sourceT = context.unsqueeze(3)
        # --> batch x idf x sourceL
        sourceT = self.conv_context(sourceT).squeeze(3)
        # seg [batch, P_NUM, queryL]
        # sourceT [batch, 32, sourceL]
        seg = seg.view(batch_size, cfg.GAN.P_NUM, queryL)
        max_seg = torch.zeros([batch_size, queryL])
        for i in range(cfg.GAN.P_NUM):
            nonzero = torch.nonzero(seg[:, i, :])
            max_seg[nonzero[:, 0], nonzero[:, 1]] = i + 1
        max_seg = max_seg.reshape(-1).int()
        part2attr_mask = torch.from_numpy(part2attr_np[max_seg, :]).cuda()
        attn = part2attr_mask.float()
        # --> batch x queryL x sourceL
        attn = attn.view(batch_size, queryL, sourceL)
        # --> batch x sourceL x queryL
        attn = torch.transpose(attn, 1, 2).contiguous() #[batch_size, sourceL, queryL]
        weightedContext = torch.bmm(sourceT, attn)
        weightedContext = weightedContext.view(batch_size, -1, ih, iw)
        attn = attn.view(batch_size, -1, ih, iw)
        return weightedContext, attn

class GlobalAttentionNoPart(nn.Module):
    def __init__(self, idf, cdf):
        super(GlobalAttentionNoPart, self).__init__()
        self.conv_context = conv1x1(cdf, idf)
        self.phi_key = conv1x1_1d(idf, idf)
        self.phi_value = conv1x1_1d(idf, idf)
        self.out_conv = conv1x1_1d(idf * 2, idf)

    def forward(self, input, context, seg, opt):
        """
            input: batch x idf x ih x iw (queryL=ihxiw)
            context: batch x cdf x sourceL
        """
        part2attr_np = np.ones([20, 11])
        part2attr_np[0, :] = 0

        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context.size(0), context.size(2)

        # --> batch x idf x queryL
        target = input.view(batch_size, -1, queryL)
        # --> batch x queryL x idf
        targetT = torch.transpose(target, 1, 2).contiguous()
        # batch x cdf x sourceL --> batch x cdf x sourceL x 1
        sourceT = context.unsqueeze(3)
        # --> batch x idf x sourceL
        sourceT = self.conv_context(sourceT).squeeze(3)
        # seg [batch, P_NUM, queryL]
        # sourceT [batch, 32, sourceL]
        seg = seg.view(batch_size, cfg.GAN.P_NUM, queryL)
        max_seg = torch.zeros([batch_size, queryL])
        for i in range(cfg.GAN.P_NUM):
            nonzero = torch.nonzero(seg[:, i, :])
            max_seg[nonzero[:, 0], nonzero[:, 1]] = i + 1
        max_seg = max_seg.reshape(-1).int()
        part2attr_mask = torch.from_numpy(part2attr_np[max_seg, :]).cuda()
        part2attr_mask = part2attr_mask.byte()
        cnt_attr_tensor = part2attr_mask.sum(dim=1)
        softmax_index = torch.nonzero(cnt_attr_tensor > 1)[:, 0]
        assert torch.nonzero(cnt_attr_tensor == 1)[:, 0].shape[0] == 0
        zero_index = torch.nonzero(cnt_attr_tensor == 0)[:, 0]

        attn = torch.bmm(targetT, sourceT)
        attn = attn.view(batch_size*queryL, sourceL)
        attn[softmax_index, :] = attn[softmax_index, :].data.masked_fill_(1 - part2attr_mask[softmax_index, :],-float('inf'))
        attn[softmax_index, :] = nn.Softmax()(attn[softmax_index, :])        # --> (batch*queryL, sourceL)
        attn[zero_index, :] = 0

        # --> batch x queryL x sourceL
        attn = attn.view(batch_size, queryL, sourceL)
        # --> batch x sourceL x queryL
        attn = torch.transpose(attn, 1, 2).contiguous() #[batch_size, sourceL, queryL]
        weightedContext = torch.bmm(sourceT, attn)
        weightedContext = weightedContext.view(batch_size, -1, ih, iw)
        attn = attn.view(batch_size, -1, ih, iw)

        return weightedContext, attn
