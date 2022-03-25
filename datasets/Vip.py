import os
import io
import pickle
import torch
import numpy as np
import random
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from miscc.config import cfg
from miscc.manu_data import vip_attr_list, vip_split_attr, vip_attr_keys
from miscc.utils import load_acts_data, load_bytes_data, get_activations
from models.Pmodel import INCEPTION_V3_FID

def prepare_train_data(data):
    imgs, segs, pooled_segs, att, class_id, acts, filenames = data
    real_imgs = []
    for i in range(len(imgs)):
        real_imgs.append(Variable(imgs[i]).cuda())

    real_segs, real_pooled_segs = [], []
    for i in range(len(segs)):
        real_segs.append(Variable(segs[i].float()).cuda())
        real_pooled_segs.append(Variable(pooled_segs[i].float()).cuda())
    att = Variable(att).cuda()
    class_id = class_id.numpy()
    acts = acts.numpy()
    return [real_imgs, real_segs, real_pooled_segs, att, class_id, acts, filenames]

def prepare_acts_data(data):
    imgs, names = data
    real_imgs = []
    for i in range(len(imgs)):
        real_imgs.append(Variable(imgs[i]).cuda())

    return [real_imgs, names]

class LabelDataset(data.Dataset):
    def __init__(self, transform, split):
        self.split = split
        self.vip_attr_num = len(vip_attr_list)
        self.vip_split_num = len(vip_split_attr)
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.imsize = []
        base_size_width = cfg.TREE.BASE_SIZE_WIDTH
        base_size_height = cfg.TREE.BASE_SIZE_HEIGHT
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append([base_size_height, base_size_width])
            base_size_height *= 2
            base_size_width *= 2
        if split == 'train': self.batch_size = cfg.TRAIN.BATCH_SIZE
        elif split == 'test': self.batch_size = cfg.TEST.BATCH_SIZE
        
        att_path = os.path.join(cfg.DATA_DIR, 'att_anno.pkl')
        with open(att_path, 'rb') as f:
            self.att_dict = pickle.load(f)
        self.video_inds = self.load_video_inds()
        self.filenames = self.load_filenames()
        self.class_ids = self.load_class_ids()
        self.atts = self.process_atts()

        self.img_bytes = load_bytes_data(split, self.filenames, 'imgs', '.png')
        self.seg_bytes = load_bytes_data(split, self.filenames, 'segs', '.png')

        self.acts_dict = self.load_acts()

    def load_video_inds(self):
        filepath = os.path.join(cfg.DATA_DIR, self.split, 'videos.txt')
        with open(filepath, 'rb') as f:
            video_inds = f.readlines()

        video_inds = [int(ind.rstrip()[6:]) for ind in video_inds]
        return video_inds

    def load_filenames(self):
        filepath = os.path.join(cfg.DATA_DIR, self.split, 'filenames.pickle')
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
            for key in self.att_dict:
                video_name, class_name, frame_name, human_fake_id = key.split('_')
                video_index = int(video_name[6:])
                if video_index not in self.video_inds:
                    continue
                filename = os.path.join(video_name, class_name, frame_name + '_' + human_fake_id)
                img_path = os.path.join(cfg.DATA_DIR, 'imgs', '{}.png'.format(filename))
                seg_path = os.path.join(cfg.DATA_DIR, 'segs', '{}.png'.format(filename))
                if os.path.exists(img_path) and os.path.exists(seg_path):
                    filenames.append(filename)

            with open(filepath, 'wb') as f:
                pickle.dump(filenames, f)
                print('Save to: ', filepath)

        return filenames

    def load_class_ids(self):
        filepath = os.path.join(cfg.DATA_DIR, self.split, 'class_ids.pickle')
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                class_ids = pickle.load(f)
            print('Load class_ids from: %s (%d)' % (filepath, len(class_ids)))
        else:
            class_ids = []
            class_id_dict = {}
            cnt_id = 1
            for key in self.att_dict:
                video_name, class_name, frame_name, human_fake_id = key.split('_')
                video_index = int(video_name[6:])

                if video_index not in self.video_inds:
                    continue

                class_id = video_name + '_' + class_name
                if class_id not in class_id_dict:
                    class_id_dict[class_id] = cnt_id
                    cnt_id = cnt_id + 1
                class_ids.append(class_id_dict[class_id])

            with open(filepath, 'wb') as f:
                pickle.dump(class_ids, f)
                print('Save to: ', filepath)

        return class_ids

    def process_atts(self):
        filepath = os.path.join(cfg.DATA_DIR, self.split, 'atts.pickle')
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                atts = pickle.load(f)
            print('Load att_sets from: %s (%d)' % (filepath, len(atts)))
        else:
            atts = []
            for key in self.att_dict:
                video_name, class_name, frame_name, human_fake_id = key.split('_')
                video_index = int(video_name[6:])

                if video_index not in self.video_inds:
                    continue

                raw_att = self.att_dict[key]
                att = np.zeros([self.vip_split_num, self.vip_attr_num])
                for i, att_name in enumerate(vip_attr_keys):
                    label_indexes = raw_att[att_name].split(',')
                    for label_index in label_indexes:
                        label_index = int(label_index)
                        att[i, vip_split_attr[i] + label_index] = 1
                atts.append(att)

            with open(filepath, 'wb') as f:
                pickle.dump(atts, f)
                print('Save to: ', filepath)

        return atts

    def process_imgs(self, img_byte, seg_byte):
        img = Image.open(io.BytesIO(img_byte)).convert('RGB')
        seg = Image.open(io.BytesIO(seg_byte))

        if self.transform is not None:
            img, seg = self.transform(img, seg)

        ret = []
        new_segs = []
        pooled_segs = []

        for i in range(cfg.TREE.BRANCH_NUM):
            if i < (cfg.TREE.BRANCH_NUM - 1):
                re_img = transforms.Resize(self.imsize[i])(img)
                re_seg = transforms.Resize(self.imsize[i], Image.NEAREST)(seg)
            else:
                re_img = img
                re_seg = seg
            ret.append(self.norm(re_img))
            re_seg = np.asarray(re_seg)
            new_seg = np.zeros([cfg.GAN.P_NUM, self.imsize[i][0], self.imsize[i][1]])

            for j in range(cfg.GAN.P_NUM):
                if j == 12: continue # face
                new_seg[j, re_seg == (j + 1) * 10] = 1
            pooled_seg = np.amax(new_seg, axis=0)
            pooled_segs.append(pooled_seg)
            new_segs.append(new_seg)

        return ret, new_segs, pooled_segs

    def dump_fid_acts(self, filepath):
        incep_path = os.path.join(cfg.PRETRAINED_DIR, 'inception_v3_google-1a9a5a14.pth')
        incep_state_dict = torch.load(incep_path, map_location=lambda storage, loc: storage)

        block_idx = INCEPTION_V3_FID.BLOCK_INDEX_BY_DIM[cfg.TEST.FID_DIMS]
        inception_model_fid = INCEPTION_V3_FID(incep_state_dict, [block_idx])
        inception_model_fid.cuda()
        inception_model_fid = nn.DataParallel(inception_model_fid)
        inception_model_fid.eval()
        act_dataset = create_acts_dataset(self.split, self.img_bytes, self.filenames)
        act_dataloader = torch.utils.data.DataLoader(
            act_dataset, batch_size=self.batch_size, drop_last=False, 
            shuffle=False, num_workers=int(cfg.WORKERS))
        acts_dict = {}
        count = 0
        for step, data in enumerate(act_dataloader):
            if count % 10 == 0:
                print('%07d / %07d'%(count, self.__len__() / self.batch_size))
            imgs, names = prepare_acts_data(data)
            batch_size = len(names)
            acts = get_activations(imgs[-1], inception_model_fid, batch_size)
            for batch_index in range(batch_size):
                acts_dict[names[batch_index]] = acts[batch_index]

            count += 1
        with open(filepath, 'wb') as f:
            pickle.dump(acts_dict, f)
            print('Save to: ', filepath)

        return acts_dict

    def load_acts(self):
        filepath = os.path.join(cfg.DATA_DIR, self.split, 'acts.pickle')
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                acts_dict = pickle.load(f)
            print('Load acts_dict from: %s (%d)' % (filepath, len(acts_dict)))
        else:
            acts_dict = self.dump_fid_acts(filepath)

        return acts_dict

    def __getitem__(self, index):
        imgs, segs, pooled_segs = self.process_imgs(self.img_bytes[index], self.seg_bytes[index])
        att = self.atts[index].astype('float32')
        class_id = self.class_ids[index]
        acts = self.acts_dict[self.filenames[index]]
        return imgs, segs, pooled_segs, att, class_id, acts, self.filenames[index]

    def __len__(self):
        return len(self.filenames)

class create_acts_dataset(data.Dataset):
    def __init__(self, split, img_bytes, filenames):
        self.split = split
        self.transform = transforms.Compose([
            transforms.Resize(
                (int(cfg.TREE.BASE_SIZE_HEIGHT * (2 ** (cfg.TREE.BRANCH_NUM - 1))),
                 int(cfg.TREE.BASE_SIZE_WIDTH * (2 ** (cfg.TREE.BRANCH_NUM - 1))))),
        ])
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.imsize = []
        base_size_width = cfg.TREE.BASE_SIZE_WIDTH
        base_size_height = cfg.TREE.BASE_SIZE_HEIGHT
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append([base_size_height, base_size_width])
            base_size_height *= 2
            base_size_width *= 2
        self.img_bytes = img_bytes
        self.filenames = filenames

    def get_imgs(self, img_byte):
        img = Image.open(io.BytesIO(img_byte)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        ret = []
        for i in range(cfg.TREE.BRANCH_NUM):
            if i < (cfg.TREE.BRANCH_NUM - 1):
                re_img = transforms.Resize(self.imsize[i])(img)
            else:
                re_img = img
            ret.append(self.norm(re_img))
        return ret

    def __getitem__(self, index):
        imgs = self.get_imgs(self.img_bytes[index])
        return imgs, self.filenames[index]

    def __len__(self):
        return len(self.filenames)