from __future__ import print_function
from six.moves import range

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from PIL import Image
import numpy as np
from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import weights_init, load_params, copy_G_params
from miscc.utils import calculate_activation_statistics, calculate_frechet_distance
from miscc.utils import get_activations, compute_inception_score, negative_log_posterior_probability
from models.Rmodel import G_NET
from models.Emodel import LabelEncoder, CNN_ENCODER
from models.Pmodel import INCEPTION_V3, INCEPTION_V3_FID
from miscc.losses import words_loss
from miscc.utils import build_images, prepare_condition, save_single_img, save_every_single_img
from datasets.Vip import prepare_train_data
import os
import time
import pickle

# ################# Text to image task############################ #
class condGANTester(object):
    def __init__(self, output_dir, test_dataloader, test_dataset):
        self.model_dir = os.path.join(output_dir, 'Model')
        self.image_dir = os.path.join(output_dir, 'Image')
        self.score_dir = os.path.join(output_dir, 'Score')
        mkdir_p(self.model_dir)
        mkdir_p(self.image_dir)
        mkdir_p(self.score_dir)
        if len(cfg.GPU_IDS) == 1 and cfg.GPU_IDS[0] >= 0:
            torch.cuda.set_device(0)
        cudnn.benchmark = True
        self.test_dataset = test_dataset
        self.vip_attr_num = test_dataset.vip_attr_num
        self.vip_split_num = test_dataset.vip_split_num
        self.test_dataloader = test_dataloader
        self.num_batches = len(self.test_dataloader)

    def build_models(self, states):
        # ###################encoders######################################## #
        if cfg.TRAIN.NET_LE == '' or cfg.TRAIN.NET_IE == '':
            print('Error: no pretrained label-image encoders')
            return

        incep_path = os.path.join(cfg.PRETRAINED_DIR, 'inception_v3_google-1a9a5a14.pth')
        incep_state_dict = torch.load(incep_path, map_location=lambda storage, loc: storage)

        image_encoder = CNN_ENCODER(cfg.IMAGE.EMBEDDING_DIM, incep_state_dict).cuda()
        image_encoder = nn.DataParallel(image_encoder, device_ids=cfg.GPU_IDS)
        img_encoder_path = os.path.join(cfg.PRETRAINED_DIR, cfg.TRAIN.NET_IE)
        state_dict = torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
        image_encoder.load_state_dict(state_dict)
        for p in image_encoder.parameters():
            p.requires_grad = False
        print('Load image encoder from:', img_encoder_path)
        image_encoder.eval()

        label_encoder = LabelEncoder(self.vip_attr_num, self.vip_split_num).cuda()
        label_encoder = nn.DataParallel(label_encoder, device_ids=cfg.GPU_IDS)
        txt_encoder_path = os.path.join(cfg.PRETRAINED_DIR, cfg.TRAIN.NET_LE)
        state_dict = torch.load(txt_encoder_path, map_location=lambda storage, loc: storage)
        label_encoder.load_state_dict(state_dict)
        for p in label_encoder.parameters():
            p.requires_grad = False
        print('Load text encoder from:', txt_encoder_path)
        label_encoder.eval()

        # ####################### generator ############## #
        assert states is not None
        netG = nn.DataParallel(G_NET(),device_ids=cfg.GPU_IDS)
        netG.apply(weights_init)
        netG.cuda().eval()
        epoch = int(states['epoch'])+1
        netG.load_state_dict(states['avg_netG'])

        # #######################evaluation models############## #
        self.inception_model = INCEPTION_V3(incep_state_dict)
        self.inception_model = nn.DataParallel(self.inception_model)
        self.inception_model.cuda()
        self.inception_model.eval()
        block_idx = INCEPTION_V3_FID.BLOCK_INDEX_BY_DIM[cfg.TEST.FID_DIMS]
        self.inception_model_fid = INCEPTION_V3_FID(incep_state_dict, [block_idx])
        self.inception_model_fid = nn.DataParallel(self.inception_model_fid)
        self.inception_model_fid.cuda()
        self.inception_model_fid.eval()
        return [label_encoder, image_encoder, netG, epoch]

    def prepare_labels(self, batch_size):
        match_labels = Variable(torch.LongTensor(range(batch_size))).cuda()
        label_tensor = torch.Tensor(range(self.vip_split_num)).long().cuda()
        label_tensor = label_tensor.expand(len(cfg.GPU_IDS), self.vip_split_num)
        return label_tensor, match_labels

    def save_img_results(self, netG, noise, att_embs, att, epoch, imgs, segs, pooled_segs):
        # Save images
        raw_fake_img, fake_img_front, attn_maps = netG(noise, att_embs, segs, pooled_segs, imgs)
        back_index = (pooled_segs[-1] == 0).unsqueeze(dim=1).expand_as(imgs[-1])
        front_index = (pooled_segs[-1] == 1).unsqueeze(dim=1).expand_as(imgs[-1])
        fake_img_background = torch.zeros(fake_img_front.shape)
        fake_img_background[back_index] = imgs[-1][back_index].cpu().detach()
        fake_img_background[front_index] = fake_img_front[front_index].cpu().detach()  # [batch_size, channel, height, width]
        seg = segs[-1].cpu()
        attn_map = attn_maps[-1]
        build_images(fake_img_background, imgs[-1], att, attn_map, seg, self.vip_split_num, epoch, self.image_dir)

        back_index = (pooled_segs[-1] == 0).unsqueeze(dim=1).expand_as(imgs[-1])
        front_index = (pooled_segs[-1] == 1).unsqueeze(dim=1).expand_as(imgs[-1])
        raw_fake_img_background = torch.zeros(raw_fake_img[-1].shape)
        raw_fake_img_background[back_index] = imgs[-1][back_index].cpu().detach()
        raw_fake_img_background[front_index] = raw_fake_img[-1][front_index].cpu().detach()  # [batch_size, channel, height, width]
        seg = segs[-1].cpu()
        attn_map = attn_maps[-1]
        build_images(raw_fake_img_background, imgs[-1], att, attn_map, seg, self.vip_split_num, epoch, self.image_dir, raw=True)

    def test(self):
        states = None
        if cfg.CKPT != '':
            print('Load CKPT from: ', cfg.CKPT)
            states = torch.load(cfg.CKPT, map_location=lambda storage, loc: storage)

        label_encoder, image_encoder, netG, epoch = self.build_models(states)

        label_tensor, match_labels = self.prepare_labels(cfg.TEST.BATCH_SIZE)
        batch_size = cfg.TEST.BATCH_SIZE
        nz = cfg.GAN.Z_DIM
        predictions, fake_acts_set, acts_set, w_accuracy = [], [], [], []
        for step, data in enumerate(self.test_dataloader, 0):
            noise = Variable(torch.FloatTensor(batch_size, nz))
            noise = noise.cuda()
            noise = noise.data.normal_(0,1)
            ######################################################
            # (1) Prepare training data and Compute text embeddings
            imgs, segs, pooled_segs, att, class_ids, acts, filenames = prepare_train_data(data)
            batch_size = len(class_ids)
            # att_embs: batch_size x nef x seq_len
            att_embs, _ = label_encoder(label_tensor, att)
            att_embs = att_embs.detach()
            #######################################################
            # (2) Generate fake images
            raw_fake_imgs, fake_img_front, _ = netG(noise[:batch_size], att_embs, segs, pooled_segs, imgs)
            back_index = (pooled_segs[-1] == 0).unsqueeze(dim=1).expand_as(imgs[-1])
            front_index = (pooled_segs[-1] == 1).unsqueeze(dim=1).expand_as(imgs[-1])
            fake_img_background = torch.zeros(fake_img_front.shape).cuda()
            fake_img_background[back_index] = imgs[-1][back_index]
            fake_img_background[front_index] = fake_img_front[front_index]  # [batch_size, channel, height, width]
            # fake_img_background = raw_fake_imgs[-1]
            ### score ################################################
            images = fake_img_background
            region_features, cnn_code = image_encoder(images)
            region_features, cnn_code = region_features.detach(), cnn_code.detach()
            _, _, _, w_accu = words_loss(region_features, segs, att_embs, match_labels[:batch_size], self.vip_split_num, class_ids, batch_size)
            w_accuracy.append(w_accu)
            pred = self.inception_model(images)
            pred = pred.data.cpu().numpy()
            predictions.append(pred)
            fake_acts = get_activations(images, self.inception_model_fid, batch_size)
            acts_set.append(acts)
            fake_acts_set.append(fake_acts)

            ### save
            save_every_single_img(self.image_dir, filenames, images, pooled_segs, imgs)

            if step % cfg.TEST.PRINT_INTERVAL == 0:
                print('[Iter {0}/{1}]'.format(step, self.num_batches))

        accu_w, std_w = np.mean(w_accuracy), np.std(w_accuracy)
        acts_set = np.concatenate(acts_set, 0)
        fake_acts_set = np.concatenate(fake_acts_set, 0)
        real_mu, real_sigma = calculate_activation_statistics(acts_set)
        fake_mu, fake_sigma = calculate_activation_statistics(fake_acts_set)
        fid_score = calculate_frechet_distance(real_mu, real_sigma, fake_mu, fake_sigma)

        print('score: accu_w, std_w, fid_score')
        print('score: %f %f, %f' %(accu_w, std_w, fid_score))

        fullpath = os.path.join(self.score_dir, 'scores_{0}.txt'.format(epoch))
        with open(fullpath, 'w') as fp:
            fp.write('accu_w, std_w, fid_score \n')
            fp.write('%f, %f, %f' %(accu_w, std_w, fid_score))

        self.save_img_results(netG, noise[:batch_size], att_embs, att, epoch, imgs, segs, pooled_segs)