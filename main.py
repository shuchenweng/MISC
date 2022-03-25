from __future__ import print_function
import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
import torch

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

from miscc.config import cfg, cfg_from_file
import miscc.compose as transforms
from datasets.Vip import LabelDataset
from trainer import condGANTrainer as trainer
from tester import condGANTester as tester


def parse_args():
    parser = argparse.ArgumentParser(description='Train a CRAC network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/test_SC.yml', type=str)
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--gpuIDs', type=str, default='0')
    parser.add_argument('--maxEpoch', type=int, default=-1)
    parser.add_argument('--ckpt', type=str, default='')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    cfg_from_file(args.cfg_file)
    if args.maxEpoch != -1: cfg.MAX_EPOCH = args.maxEpoch
    if args.ckpt: cfg.CKPT = args.ckpt
    assert args.gpuIDs != '-1'
    cfg.GPU_IDS = [int(gpu_id) for gpu_id in range(len(args.gpuIDs.split(',')))]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuIDs
    cfg.TRAIN.BATCH_SIZE = len(cfg.GPU_IDS)*cfg.TRAIN.UNI_BATCH_SIZE
    cfg.TEST.BATCH_SIZE = len(cfg.GPU_IDS)*cfg.TEST.UNI_BATCH_SIZE

    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    torch.cuda.manual_seed_all(args.manualSeed)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = os.path.join(cfg.MODEL_DIR, cfg.CONFIG_NAME, timestamp)

    # Get data loader
    imsize_width = cfg.TREE.BASE_SIZE_WIDTH * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    imsize_height = cfg.TREE.BASE_SIZE_HEIGHT * (2 ** (cfg.TREE.BRANCH_NUM - 1))

    start_t = time.time()

    test_image_transform = transforms.Compose([
        transforms.Resize((int(imsize_height), int(imsize_width)))
    ])
    test_dataset = LabelDataset(split='test', transform=test_image_transform)
    assert test_dataset
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.TEST.BATCH_SIZE, drop_last=True, 
        shuffle=False, num_workers=int(cfg.WORKERS))

    if cfg.TRAIN.FLAG:
        train_image_transform = transforms.Compose([
            transforms.Resize((int(imsize_height * 76 / 64), int(imsize_width * 76 / 64))),
            transforms.RandomCrop((imsize_height, imsize_width)),
            transforms.RandomHorizontalFlip()])
        train_dataset = LabelDataset(split='train', transform=train_image_transform)
        assert train_dataset
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, drop_last=True,
            shuffle=True, num_workers=int(cfg.WORKERS))
        algo = trainer(output_dir, train_dataloader, train_dataset, test_dataloader, test_dataset)
        algo.train()
    else:
        train_dataset = LabelDataset(split='train', transform=test_image_transform)
        assert train_dataset
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.TEST.BATCH_SIZE, drop_last=True,
            shuffle=False, num_workers=int(cfg.WORKERS))
        algo = tester(output_dir, test_dataloader, test_dataset)
        algo.test()
    end_t = time.time()
    print('Total time for training:', end_t - start_t)