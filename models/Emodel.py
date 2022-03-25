import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from miscc.config import cfg
from models.Blocks import conv1x1

class LabelEncoder(nn.Module):
    def __init__(self, vip_attr_num, split_num):
        super(LabelEncoder, self).__init__()
        self.vip_attr_num = vip_attr_num
        self.split_num = split_num
        self.label_emb_num = cfg.TRAIN.LABEL_EMB_NUM
        self.attr_emb_num = cfg.TRAIN.ATT_EMB_NUM
        self.define_module()
        self.init_trainable_weights()
        self.beta = cfg.TRAIN.SMOOTH.BETA

    def init_trainable_weights(self):
        initrange = 0.1
        self.label_embedding.weight.data.uniform_(-initrange, initrange)
        self.attr_embedding.weight.data.uniform_(-initrange, initrange)
        for layer in self.hidden_layers:
            if not isinstance(layer, nn.Linear): continue
            layer.weight.data.uniform_(-initrange, initrange)

    def define_module(self):
        self.label_embedding = nn.Embedding(self.split_num, self.label_emb_num)
        self.attr_embedding = nn.Linear(self.vip_attr_num, self.attr_emb_num)
        self.s_emb_num = self.attr_emb_num + self.label_emb_num
        self.hidden_layers = nn.Sequential(
            nn.Linear(self.s_emb_num, self.s_emb_num//2),
            nn.ReLU(),
            nn.Linear(self.s_emb_num//2, self.s_emb_num//4),
        )

    def forward(self, label, attr):
        batch_size = attr.size(0)
        label_emb = self.label_embedding(label) #[1, vip_split]

        label_emb = label_emb.expand((batch_size, self.split_num, self.label_emb_num))#[batchsize, vip_split, emb_dim]
        label_emb = label_emb.contiguous().view(batch_size * self.split_num, self.label_emb_num)

        attr = attr.view(batch_size * self.split_num, self.vip_attr_num)
        attr_emb = self.attr_embedding(attr)

        concat_emb = torch.cat([label_emb, self.beta * attr_emb], 1)
        for layer in self.hidden_layers:
            concat_emb = layer(concat_emb)
        concat_emb = concat_emb.view(batch_size, self.split_num, self.s_emb_num//4)
        global_emb = concat_emb.mean(1)
        return concat_emb.permute((0,2,1)), global_emb

class CNN_ENCODER(nn.Module):
    def __init__(self, nef, incep_state_dict):
        super(CNN_ENCODER, self).__init__()
        self.nef = nef
        model = models.inception_v3()
        model.load_state_dict(incep_state_dict)
        for param in model.parameters():
            param.requires_grad = False

        self.define_module(model)
        self.init_trainable_weights()

    def define_module(self, model):
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c

        self.emb_features = conv1x1(768, self.nef)
        self.emb_cnn_code = nn.Linear(2048, self.nef)

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        features = None
        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.Upsample(size=(299, 299), mode='bilinear')(x)
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192

        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288

        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768

        # image region features
        features = x
        # 17 x 17 x 768

        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        # x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048

        # global image features
        cnn_code = self.emb_cnn_code(x)
        # 512
        if features is not None:
            features = self.emb_features(features)
        return features, cnn_code