import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from miscc.manu_data import vip_split_attr, attr2part_dict, part2attr_np
from miscc.utils import get_pt_version
from miscc.config import cfg
# PT_VERSION = get_pt_version()


# ##################Loss for matching text-image###################
def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

def func_attention(query, context, segs):
    """
    query: batch x ndf x queryL
    context: batch x ndf x ih x iw (sourceL=ihxiw)
    mask: batch_size x sourceL
    """
    batch_size, queryL = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    sourceL = ih * iw

    # --> batch x sourceL x ndf
    context = context.view(batch_size, -1, sourceL)
    seg = nn.Upsample(size=(ih, iw), mode='nearest')(segs[-1])
    seg = seg.view(batch_size, cfg.GAN.P_NUM, sourceL)
    max_seg = torch.zeros(batch_size, sourceL)
    for i in range(cfg.GAN.P_NUM):
        nonzero = torch.nonzero(seg[:, i, :])
        max_seg[nonzero[:,0], nonzero[:,1]] = i + 1
    max_seg = max_seg.reshape(-1).int()
    part2attr_mask = torch.from_numpy(part2attr_np[max_seg, :]).cuda()
    attn = part2attr_mask.float()
    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size*queryL, sourceL)
    #  Eq. (9)
    attr2part_mask = torch.zeros([batch_size, queryL, sourceL]).cuda()
    for i in range(queryL):
        attr2part_mask[:, i, :] = seg[:, attr2part_dict[i] - 1, :].sum(dim=1)
    attr2part_mask = attr2part_mask.view(batch_size*queryL, sourceL)
    # if PT_VERSION >= 120:
    attr2part_mask = attr2part_mask.bool()
    attn = attn.data.masked_fill_(attr2part_mask.bitwise_not(), -float('inf'))
    # else:
    #     attr2part_mask = attr2part_mask.byte()
    #     attn = attn.data.masked_fill_(1 - attr2part_mask, -float('inf'))

    attn = nn.Softmax(dim=1)(attn)
    attn = attn.view(batch_size, queryL, sourceL)
    attn[torch.isnan(attn)] = 0

    # # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(context, attnT)

    return weightedContext, attn.view(batch_size, -1, ih, iw)

def words_loss(img_features, segs, att_embs, labels, words_num, class_ids, batch_size):
    """
        att_embs(query): batch x nef x seq_len
        img_features(context): batch x nef x 17 x 17
    """
    masks = []
    attn_maps = []
    similarities = []
    for i in range(batch_size):
        if class_ids is not None:
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        # Get the i-th text description
        # -> 1 x nef x words_num
        word = att_embs[i, :, :words_num].unsqueeze(0).contiguous()
        # -> batch_size x nef x words_num
        word = word.repeat(batch_size, 1, 1)
        # batch x nef x 17*17
        context = img_features
        """
            word(query): batch x nef x words_num
            context: batch x nef x 17 x 17
            weiContext: batch x nef x words_num
            attn: batch x words_num x 17 x 17
        """
        weiContext, attn = func_attention(word, context, segs)
        attn_maps.append(attn[i].unsqueeze(0).contiguous())
        # --> batch_size x words_num x nef
        word = word.transpose(1, 2).contiguous()
        weiContext = weiContext.transpose(1, 2).contiguous()
        # --> batch_size*words_num x nef
        word = word.view(batch_size * words_num, -1)
        weiContext = weiContext.view(batch_size * words_num, -1)
        #
        # -->batch_size*words_num
        row_sim = cosine_similarity(word, weiContext)
        # --> batch_size x words_num
        row_sim = row_sim.view(batch_size, words_num)

        # Eq. (10)pa
        row_sim.mul_(cfg.TRAIN.SMOOTH.GAMMA2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)

        # --> 1 x batch_size
        # similarities(i, j): the similarity between the i-th image and the j-th text description
        similarities.append(row_sim)

    # batch_size x batch_size
    similarities = torch.cat(similarities, 1)
    if class_ids is not None:
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        # if PT_VERSION >= 120:
        masks = torch.BoolTensor(masks).cuda()
        # else:
        #     masks = torch.ByteTensor(masks).cuda()

    similarities = similarities * cfg.TRAIN.SMOOTH.GAMMA3
    if class_ids is not None:
        similarities.data.masked_fill_(masks, -float('inf'))
    similarities1 = similarities.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(similarities, labels)
        loss1 = nn.CrossEntropyLoss()(similarities1, labels)
        _, predicted = torch.max(similarities, 1)
        _, predicted1 = torch.max(similarities1, 1)
        correct = ((predicted == labels).sum().cpu().item() + (predicted1 == labels).sum().cpu().item())
        accuracy = (100. * correct) / (batch_size * 2.)
    else:
        loss0, loss1 = None, None
    return loss0, loss1, attn_maps, accuracy

# ##################Loss for G and Ds##############################
def patD_loss(netPatD, real_imgs, fake_img, conditions):
    # Forward
    batch_size = real_imgs.size(0)
    real_features = netPatD(real_imgs)
    fake_features = netPatD(fake_img.detach())
    # loss
    cond_real_logits = netPatD.module.COND_DNET(real_features, conditions)
    cond_fake_logits = netPatD.module.COND_DNET(fake_features, conditions)
    real_labels = Variable(torch.FloatTensor(cond_real_logits.size()).fill_(1)).cuda()
    fake_labels = Variable(torch.FloatTensor(cond_fake_logits.size()).fill_(0)).cuda()
    cond_real_errD = nn.BCELoss()(cond_real_logits, real_labels)
    cond_fake_errD = nn.BCELoss()(cond_fake_logits, fake_labels)
    #
    cond_wrong_logits = netPatD.module.COND_DNET(real_features[:(batch_size - 1)], conditions[1:batch_size])
    cond_wrong_errD = nn.BCELoss()(cond_wrong_logits, fake_labels[1:batch_size])

    real_logits = netPatD.module.UNCOND_DNET(real_features)
    fake_logits = netPatD.module.UNCOND_DNET(fake_features)
    real_errD = nn.BCELoss()(real_logits, real_labels)
    fake_errD = nn.BCELoss()(fake_logits, fake_labels)
    errD = (real_errD / 2. + fake_errD / 3.) * cfg.TRAIN.SMOOTH.LAMBDA2 +\
           (cond_real_errD / 2. + cond_fake_errD / 3. + cond_wrong_errD / 3.)
    return errD

def segD_loss(netSegD, fake_img_background, pool_seg):
    fake_seg_result = netSegD(fake_img_background.detach())
    seg_groundtruth = pool_seg.unsqueeze(dim=1)
    errD = nn.BCELoss()(fake_seg_result, seg_groundtruth)
    return errD

def G_loss(netsPatD, netSegD, image_encoder, raw_fake_imgs, fake_img_background, 
    segs, pooled_segs, att_embs, condition, match_labels, class_ids):
    numDs = len(netsPatD)
    batch_size = fake_img_background.size(0)
    height = cfg.TREE.BASE_SIZE_HEIGHT
    width = cfg.TREE.BASE_SIZE_WIDTH
    logs = ''
    # Forward
    errG_total = 0
    for i in range(numDs):
        features = netsPatD[i](raw_fake_imgs[i])

        sample_condition = nn.Upsample(size=(height*(2**i)//16, width*(2**i)//16), mode='nearest')(condition)
        cond_logits = netsPatD[i].module.COND_DNET(features, sample_condition)
        real_labels = Variable(torch.FloatTensor(cond_logits.size()).fill_(1)).cuda()
        cond_errG = nn.BCELoss()(cond_logits, real_labels)
        logits = netsPatD[i].module.UNCOND_DNET(features)
        errG = nn.BCELoss()(logits, real_labels)
        g_loss = errG * cfg.TRAIN.SMOOTH.LAMBDA2 + cond_errG

        if i == (numDs - 1):
            features = netsPatD[i](fake_img_background)
            sample_condition = nn.Upsample(size=(height * (2 ** i) // 16, width * (2 ** i) // 16), mode='nearest')(condition)
            cond_logits = netsPatD[i].module.COND_DNET(features, sample_condition)
            real_labels = Variable(torch.FloatTensor(cond_logits.size()).fill_(1)).cuda()
            fake_cond_errG = nn.BCELoss()(cond_logits, real_labels)
            logits = netsPatD[i].module.UNCOND_DNET(features)
            fake_errG = nn.BCELoss()(logits, real_labels)
            g_loss += fake_errG * cfg.TRAIN.SMOOTH.LAMBDA2 + fake_cond_errG

        errG_total += g_loss

        logs += 'errPatG%d: %.2f ' % (i, g_loss.item())

        # Ranking loss
        if i == (numDs - 1):
            # words_features: batch_size x nef x 17 x 17
            # sent_code: batch_size x nef
            region_features, cnn_code = image_encoder(raw_fake_imgs[i])
            w_loss0, w_loss1, _, _ = words_loss(region_features, segs, att_embs, match_labels, len(vip_split_attr), class_ids, batch_size)
            w_loss = (w_loss0 + w_loss1) * cfg.TRAIN.SMOOTH.LAMBDA
            errG_total += w_loss
            logs += 'w_loss: %.2f ' % (w_loss.item())

            front_index = (pooled_segs[i] == 1).unsqueeze(dim=1).expand_as(fake_img_background)
            color_l1_loss = (fake_img_background[front_index] - raw_fake_imgs[i][front_index]).abs().sum() / front_index.sum()
            errG_total += cfg.TRAIN.SMOOTH.LAMBDA4 * color_l1_loss
            logs += 'colorG_loss%d: %.2f ' % (i, color_l1_loss.item())

            fake_seg_result = netSegD(fake_img_background)
            fake_groundtruth = torch.zeros(fake_seg_result.shape).cuda()
            segG_loss = nn.BCELoss()(fake_seg_result, fake_groundtruth)
            errG_total += cfg.TRAIN.SMOOTH.LAMBDA3 * segG_loss
            logs += 'errSegG%d: %.2f ' % (i, segG_loss.item())

    return errG_total, logs


##################################################################
def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD
