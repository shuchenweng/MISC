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
from miscc.utils import weights_init, load_params, copy_G_params, trans_dict, CKPT
from miscc.utils import calculate_activation_statistics, calculate_frechet_distance
from miscc.utils import get_activations, compute_inception_score, negative_log_posterior_probability
from models.Rmodel import G_NET
from models.Emodel import LabelEncoder, CNN_ENCODER
from models.Pmodel import INCEPTION_V3, INCEPTION_V3_FID
from miscc.losses import patD_loss, segD_loss, G_loss, KL_loss, words_loss
from miscc.utils import build_images, prepare_condition
from datasets.Vip import prepare_train_data
import os
import time

# ################# Text to image task############################ #
class condGANTrainer(object):
    def __init__(self, output_dir, train_dataloader, train_dataset, test_dataloader, test_dataset):
        self.model_dir = os.path.join(output_dir, 'Model')
        self.image_dir = os.path.join(output_dir, 'Image')
        self.score_dir = os.path.join(output_dir, 'Score')
        mkdir_p(self.model_dir)
        mkdir_p(self.image_dir)
        mkdir_p(self.score_dir)
        if len(cfg.GPU_IDS) == 1 and cfg.GPU_IDS[0] >= 0:
            torch.cuda.set_device(0)
        cudnn.benchmark = True
        self.train_dataset = train_dataset
        self.vip_attr_num = train_dataset.vip_attr_num
        self.vip_split_num = train_dataset.vip_split_num
        self.train_dataloader = train_dataloader
        self.train_num_batches = len(self.train_dataloader)
        self.test_dataloader = test_dataloader
        self.test_num_batches = len(self.test_dataloader)
        self.ckpt_lst = []

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

        # #######################generator and discriminators############## #
        from models.Dmodel import PAT_D_NET64, PAT_D_NET128, PAT_D_NET256
        from models.Dmodel import SEG_D_NET
        netG = nn.DataParallel(G_NET(),device_ids=cfg.GPU_IDS)
        netSegD = nn.DataParallel(SEG_D_NET(), device_ids=cfg.GPU_IDS)
        netsPatD = []
        if cfg.TREE.BRANCH_NUM > 0:
            netsPatD.append(nn.DataParallel(PAT_D_NET64(), device_ids=cfg.GPU_IDS))
        if cfg.TREE.BRANCH_NUM > 1:
            netsPatD.append(nn.DataParallel(PAT_D_NET128(), device_ids=cfg.GPU_IDS))
        if cfg.TREE.BRANCH_NUM > 2:
            netsPatD.append(nn.DataParallel(PAT_D_NET256(), device_ids=cfg.GPU_IDS))
        netG.apply(weights_init)
        netG.cuda()
        netSegD.apply(weights_init)
        netSegD.cuda()
        for i in range(len(netsPatD)):
            netsPatD[i].apply(weights_init)
            netsPatD[i].cuda()
        epoch = 0
        if states is not None:
            epoch = int(states['epoch'])+1
            netG.load_state_dict(states['netG'])
            netSegD.load_state_dict(states['netSegD'])
            for i in range(len(netsPatD)):
                netsPatD[i].load_state_dict(states['netPatD{}'.format(i)])
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

        return [label_encoder, image_encoder, netG, netsPatD, netSegD, epoch]

    def define_optimizers(self, netG, netsPatD, netSegD, states):
        optimizersPatD = []
        for i in range(len(netsPatD)):
            opt = optim.Adam(netsPatD[i].parameters(),
                             lr=cfg.TRAIN.DISCRIMINATOR_LR,
                             betas=(0.5, 0.999))
            optimizersPatD.append(opt)

        optmizerSegD = optim.Adam(netSegD.parameters(),
                         lr=cfg.TRAIN.DISCRIMINATOR_LR,
                         betas=(0.5, 0.999))

        optimizerG = optim.Adam(netG.parameters(),
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))

        if states is not None:
            optimizerG.load_state_dict(states['optimizerG'])
            optmizerSegD.load_state_dict(states['optmizerSegD'])
            for i in range(len(netsPatD)):
                optimizersPatD[i].load_state_dict(states['optimizerPatD{}'.format(i)])

        return optimizerG, optimizersPatD, optmizerSegD

    def prepare_labels(self, batch_size):
        match_labels = Variable(torch.LongTensor(range(batch_size))).cuda()
        label_tensor = torch.Tensor(range(self.vip_split_num)).long().cuda()
        label_tensor = label_tensor.expand(len(cfg.GPU_IDS), self.vip_split_num)
        return label_tensor, match_labels

    def register_ckpt(self, epoch, curfid):
        ckpt_ind = None
        if len(self.ckpt_lst) < cfg.TRAIN.CKPT_UPLIMIT:
            ckpt_obj = CKPT(epoch, curfid)
            self.ckpt_lst.append(ckpt_obj)
            ckpt_ind = len(self.ckpt_lst)-1
        else:
            ind_w_maxfid = None
            ind_count = 0
            maxfid = -1
            for ckpt_obj_ in self.ckpt_lst:
                if ckpt_obj_.fid > maxfid:
                    ind_w_maxfid = ind_count
                    maxfid = ckpt_obj_.fid
                ind_count += 1
            
            if curfid < maxfid:
                name_w_maxfid = self.ckpt_lst[ind_w_maxfid].get_name()
                path_w_maxfid = os.path.join(self.model_dir, name_w_maxfid)
                os.remove(path_w_maxfid)
                rm_epoch = self.ckpt_lst[ind_w_maxfid].get_epoch()
                os.remove(os.path.join(self.image_dir, '{}.png'.format(rm_epoch)))
                os.remove(os.path.join(self.image_dir, '{}_raw.png'.format(rm_epoch)))
                print('{}.png'.format(rm_epoch))
                print('{}_raw.png'.format(rm_epoch))

                self.ckpt_lst[ind_w_maxfid].set_epoch(epoch)
                self.ckpt_lst[ind_w_maxfid].set_fid(curfid)
                ckpt_ind = ind_w_maxfid

        return ckpt_ind

    def save_model(self, netG, avg_param_G, netSegD, netsPatD, optimizerG, optmizerSegD, optimizersPatD, epoch, fid_score, ckpt_ind):
        states = {'epoch': epoch, 'fid_score': fid_score, 
            'netG': netG.state_dict(), 'optimizerG': optimizerG.state_dict(),
            'netSegD': netSegD.state_dict(), 'optmizerSegD': optmizerSegD.state_dict()}

        backup_para = copy_G_params(netG)
        load_params(netG, avg_param_G)
        states['avg_netG'] = netG.state_dict()
        load_params(netG, backup_para)

        for i in range(len(netsPatD)):
            states['netPatD{}'.format(i)] = netsPatD[i].state_dict()
            states['optimizerPatD{}'.format(i)] = optimizersPatD[i].state_dict()

        ckpt_name = self.ckpt_lst[ckpt_ind].get_name()
        ckpt_path = os.path.join(self.model_dir, ckpt_name)
        torch.save(states, ckpt_path)

        print('Save checkpoint.')

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

    def train(self):
        states = None
        if cfg.CKPT != '':
            print('Load CKPT from: ', cfg.CKPT)
            states = torch.load(cfg.CKPT, map_location=lambda storage, loc: storage)
        label_encoder, image_encoder, netG, netsPatD, netSegD, start_epoch = self.build_models(states)
        avg_param_G = copy_G_params(netG)
        optimizerG, optimizersPatD, optmizerSegD = self.define_optimizers(netG, netsPatD, netSegD, states)

        label_tensor, train_match_labels = self.prepare_labels(cfg.TRAIN.BATCH_SIZE)
        _, test_match_labels = self.prepare_labels(cfg.TEST.BATCH_SIZE)
        batch_size = cfg.TRAIN.BATCH_SIZE
        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
        gen_iterations = 0
        for epoch in range(start_epoch, cfg.TRAIN.MAX_EPOCH):
            start_t = time.time()
            for step, data in enumerate(self.train_dataloader, 0):
                ######################################################
                # (1) Prepare training data and Compute text embeddings
                imgs, segs, pooled_segs, att, class_ids, acts, filenames = prepare_train_data(data)
                batch_size = len(class_ids)
                # att_embs: batch_size x nef x seq_len
                att_embs, _ = label_encoder(label_tensor, att)
                att_embs = att_embs.detach()
                #######################################################
                # (2) Generate fake images
                noise.data.normal_(0, 1)
                raw_fake_imgs, fake_img_front, _ = netG(noise[:batch_size], att_embs, segs, pooled_segs, imgs)
                back_index = (pooled_segs[-1] == 0).unsqueeze(dim=1).expand_as(imgs[-1])
                front_index = (pooled_segs[-1] == 1).unsqueeze(dim=1).expand_as(imgs[-1])
                fake_img_background = torch.zeros(fake_img_front.shape).cuda()
                fake_img_background[back_index] = imgs[-1][back_index]
                fake_img_background[front_index] = fake_img_front[front_index]  # [batch_size, channel, height, width]
                #######################################################
                # (3) Update D network
                errD_total = 0
                errSegD_total = 0
                D_logs = ''
                # prepare condition ##################################
                height = cfg.TREE.BASE_SIZE_HEIGHT
                width = cfg.TREE.BASE_SIZE_WIDTH
                condition = prepare_condition(att_embs, segs, height, width)
                ######################################################
                for i in range(len(netsPatD)):
                    netsPatD[i].zero_grad()
                    sample_condition = nn.Upsample(size=(height*(2**i)//16, width*(2**i)//16), mode='nearest')(condition)
                    errD = patD_loss(netsPatD[i], imgs[i], raw_fake_imgs[i], sample_condition)
                    if i == (len(netsPatD) -1):
                        errD += patD_loss(netsPatD[i], imgs[i], fake_img_background, sample_condition)
                    errD.backward()
                    optimizersPatD[i].step()
                    errD_total += errD
                    D_logs += 'errPatD%d: %.2f ' % (i, errD.item())

                netSegD.zero_grad()
                errSegD = segD_loss(netSegD, fake_img_background, pooled_segs[0])
                errSegD.backward()
                optmizerSegD.step()
                errSegD_total += errSegD
                D_logs += 'errSegD: %.2f ' % errSegD.item()

                #######################################################
                # (4) Update G network: maximize log(D(G(z)))
                # compute total loss for training G
                gen_iterations += 1
                netG.zero_grad()
                errG_total, G_logs = G_loss(netsPatD, netSegD, image_encoder, raw_fake_imgs, 
                    fake_img_background, segs, pooled_segs, att_embs, condition, train_match_labels, class_ids)
                # backward and update parameters
                errG_total.backward()
                optimizerG.step()
                for p, avg_p in zip(netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                # display
                if step % cfg.TRAIN.PRINT_INTERVAL == 0:
                    print('[Epoch {0}/{1}][Iter {2}/{3}] \n'.format(epoch, cfg.TRAIN.MAX_EPOCH, 
                        step, self.train_num_batches) + D_logs + '\n' + G_logs)
            ################## evaluation after each epoch ##################
            netG.eval()
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            
            predictions, fake_acts_set, acts_set, w_accuracy = [], [], [], []
            for step, data in enumerate(self.test_dataloader, 0):
                ######################################################
                # (1) Prepare training data and Compute text embeddings
                imgs, segs, pooled_segs, att, class_ids, acts, filenames = prepare_train_data(data)
                batch_size = len(class_ids)
                # att_embs: batch_size x nef x seq_len
                att_embs, _ = label_encoder(label_tensor, att)
                att_embs = att_embs.detach()
                #######################################################
                # (2) Generate fake images
                noise.data.normal_(0, 1)
                raw_fake_imgs, fake_img_front, _ = netG(fixed_noise[:batch_size], att_embs, segs, pooled_segs, imgs)
                back_index = (pooled_segs[-1] == 0).unsqueeze(dim=1).expand_as(imgs[-1])
                front_index = (pooled_segs[-1] == 1).unsqueeze(dim=1).expand_as(imgs[-1])
                fake_img_background = torch.zeros(fake_img_front.shape).cuda()
                fake_img_background[back_index] = imgs[-1][back_index]
                fake_img_background[front_index] = fake_img_front[front_index]  # [batch_size, channel, height, width]

                ### score ################################################
                images = fake_img_background
                region_features, cnn_code = image_encoder(images)
                region_features, cnn_code = region_features.detach(), cnn_code.detach()
                _, _, _, w_accu = words_loss(region_features, segs, att_embs, test_match_labels[:batch_size], self.vip_split_num, class_ids, batch_size)
                w_accuracy.append(w_accu)
                pred = self.inception_model(images)
                pred = pred.data.cpu().numpy()
                predictions.append(pred)
                fake_acts = get_activations(images, self.inception_model_fid, batch_size)
                acts_set.append(acts)
                fake_acts_set.append(fake_acts)

                if step % cfg.TEST.PRINT_INTERVAL == 0:
                    print('[Iter {0}/{1}]'.format(step, self.test_num_batches))

                for i in range(images.shape[0]):
                    save_img = Image.fromarray(images[i].permute((1,2,0)).add(1).mul(127.5).detach().cpu().numpy().astype('uint8'))
                    filename = filenames[i].replace(cfg.PATH_SEPARATOR,'_')
                    save_img.save(os.path.join(self.image_dir, '{}.png'.format(filename)))

            accu_w, std_w = np.mean(w_accuracy), np.std(w_accuracy)
            acts_set = np.concatenate(acts_set, 0)
            fake_acts_set = np.concatenate(fake_acts_set, 0)
            real_mu, real_sigma = calculate_activation_statistics(acts_set)
            fake_mu, fake_sigma = calculate_activation_statistics(fake_acts_set)
            fid_score = calculate_frechet_distance(real_mu, real_sigma, fake_mu, fake_sigma)

            print('score: accu_w, std_w, fid_score')
            print('score: %f, %f, %f' %(accu_w, std_w, fid_score))

            fullpath = os.path.join(self.score_dir, 'scores_{0}.txt'.format(epoch))
            with open(fullpath, 'w') as fp:
                fp.write('accu_w, std_w, fid_score \n')
                fp.write('%f, %f, %f' %(accu_w, std_w, fid_score))

            end_t = time.time()

            print('''[%d/%d][%d]
                  Loss_D: %.2f Loss_G: %.2f Time: %.2fs'''
                  % (epoch, cfg.TRAIN.MAX_EPOCH, self.train_num_batches,
                     errD_total.item(), errG_total.item(),
                     end_t - start_t))

            ckpt_ind = self.register_ckpt(epoch, fid_score)
            if ckpt_ind is not None:
                self.save_model(netG, avg_param_G, netSegD, netsPatD, optimizerG, 
                    optmizerSegD, optimizersPatD, epoch, fid_score, ckpt_ind)

                self.save_img_results(netG, fixed_noise[:batch_size], att_embs,
                    att, epoch, imgs, segs, pooled_segs)

            if cfg.TRAIN.USE_MLT and ((epoch+1) % cfg.TRAIN.SNAPSHOT_INTERVAL == 0 or (epoch+1) == cfg.TRAIN.MAX_EPOCH):
                import mltracker
                snapshot_index = (epoch+1) // cfg.TRAIN.SNAPSHOT_INTERVAL
                mlt_vname = '{0}: {1:02d}'.format(cfg.CONFIG_NAME, snapshot_index)
                with mltracker.start_run():
                    mltracker.set_version(mlt_vname)
                    # mltracker.log_param("param", 5) # log parameters to be tuned
                    for ckpt_obj_ in self.ckpt_lst:
                        ckpt_name = ckpt_obj_.get_name()
                        ckpt_path = os.path.join(self.model_dir, ckpt_name)
                        mltracker.log_file(ckpt_path)
                        ckpt_epoch = ckpt_obj_.get_epoch()
                        raw_img_path = os.path.join(self.image_dir, str(ckpt_epoch) + '_raw.png')
                        mltracker.log_file(raw_img_path)
                        cmp_img_path = os.path.join(self.image_dir, str(ckpt_epoch) + '.png')
                        mltracker.log_file(cmp_img_path)

            load_params(netG, backup_para)
            netG.train()