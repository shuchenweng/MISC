import os
import errno
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from copy import deepcopy
from miscc.config import cfg
from miscc.manu_data import part2attr_dict, vip_split_attr, vip_attr_list, split_max_num
from skimage.transform.pyramids import pyramid_expand
from scipy import linalg
import pickle
import struct

def prepare_condition(att_embs, segs, height, width):
    batch_size = att_embs.size(0)
    part_emb = torch.zeros([batch_size, cfg.GAN.P_NUM + 1, cfg.TEXT.EMBEDDING_DIM * split_max_num]).cuda()  # with background
    for j in range(cfg.GAN.P_NUM + 1):
        label_index = np.asarray(part2attr_dict[j])
        length_label_index = label_index.shape[0]
        EX_EMBEDDING_DIM = torch.zeros([batch_size, cfg.TEXT.EMBEDDING_DIM * split_max_num]).cuda()
        tmp_att_emb = att_embs[:, :, label_index].view(batch_size, -1)
        EX_EMBEDDING_DIM[:, :length_label_index * cfg.TEXT.EMBEDDING_DIM] = tmp_att_emb
        part_emb[:, j, :] = EX_EMBEDDING_DIM  # with background

    condition = torch.zeros([batch_size, height * width, cfg.TEXT.EMBEDDING_DIM * split_max_num]).cuda()
    for j in range(cfg.GAN.P_NUM):
        seg = segs[0].view(batch_size, cfg.GAN.P_NUM, height * width)  # without background
        part_index = torch.nonzero(seg[:, j, :])
        condition[part_index[:, 0], part_index[:, 1], :] = part_emb[part_index[:, 0], j + 1, :]

    condition = condition.view(batch_size, height, width, cfg.TEXT.EMBEDDING_DIM * split_max_num).permute((0, 3, 1, 2))
    return condition

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten

def drawCaption(nvis, convas, att, show_size, FONT_MAX):
    img_txt = Image.fromarray(convas)
    fnt = None  # ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 50)
    d = ImageDraw.Draw(img_txt)
    sentence_list = []
    for i in range(nvis):
        sentence = []
        cap = att[i].data.cpu().numpy()
        for j in range(cap.shape[0]):
            word = ''
            float_att = att[i, j, :].data.cpu().detach()
            word_index = np.where(float_att == 1)[0]
            for k, index in enumerate(word_index):
                if k != 0: word += ','
                word += vip_attr_list[index]
            d.text(((j + 2) * show_size, i * FONT_MAX), '[%s]' % word, font=fnt, fill=(255, 255, 255, 255))
            sentence.append(word)
        sentence_list.append(sentence)
    return img_txt, sentence_list

def save_single_img(images, save_pth):
    images.add_(1).div_(2).mul_(255)
    images = images.data.cpu().numpy()
    images = np.transpose(images, (0, 2, 3, 1))
    img = images[3]
    img = Image.fromarray(np.uint8(img))
    img.save(save_pth)
    img = images[4]
    img = Image.fromarray(np.uint8(img))
    save_pth_1 = save_pth.replace('.png', '_1.png')
    img.save(save_pth_1)
    img = images[5]
    img = Image.fromarray(np.uint8(img))
    save_pth_2 = save_pth.replace('.png', '_2.png')
    img.save(save_pth_2)

def save_every_single_img(save_dir, filenames, images, pool_segs, real_imgs):
    images.add_(1).div_(2).mul_(255)
    images = images.data.cpu().numpy()
    images = np.transpose(images, (0, 2, 3, 1))

    real_imgs = real_imgs[-1]
    real_imgs.add_(1).div_(2).mul_(255)
    real_imgs = real_imgs.data.cpu().numpy()
    real_imgs = np.transpose(real_imgs, (0, 2, 3, 1))

    image_num = images.shape[0]
    for i in range(image_num):
        img = Image.fromarray(np.uint8(images[i]))
        img_dir = os.path.join(save_dir, 'fake')
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)
        filename = filenames[i].replace('\\', '-')
        save_pth = os.path.join(img_dir, filename+'.png')
        img.save(save_pth)

        seg = Image.fromarray(np.uint8(pool_segs[-1][i].cpu())*255)
        seg_dir = os.path.join(save_dir, 'seg')
        if not os.path.exists(seg_dir):
            os.mkdir(seg_dir)
        save_pth = os.path.join(seg_dir, filename+'.png')
        seg.save(save_pth)

        real_img = Image.fromarray(np.uint8(real_imgs[i]))
        real_dir = os.path.join(save_dir, 'real')
        if not os.path.exists(real_dir):
            os.mkdir(real_dir)
        save_pth = os.path.join(real_dir, filename+'.png')
        real_img.save(save_pth)


def build_images(fake_imgs, imgs, att, attn_maps, seg, att_split_num, epoch, image_dir, raw=False):
    COLOR_DIC = {0: [128, 64, 128], 1: [244, 35, 232],
                 2: [70, 70, 70], 3: [102, 102, 156],
                 4: [190, 153, 153], 5: [153, 153, 153],
                 6: [250, 170, 30], 7: [220, 220, 0],
                 8: [107, 142, 35], 9: [152, 251, 152],
                 10: [70, 130, 180], 11: [220, 20, 60],
                 12: [255, 0, 0], 13: [0, 0, 142],
                 14: [119, 11, 32], 15: [0, 60, 100],
                 16: [0, 80, 100], 17: [0, 0, 230],
                 18: [0, 0, 70], 19: [0, 0, 0]}
    nvis = min(4, fake_imgs.size(0))
    show_size = 17 * 16
    FONT_MAX = 18
    attn_size = attn_maps[0].shape[-1]
    fake_imgs, imgs = fake_imgs[:nvis], imgs[:nvis].cpu()
    fake_imgs, imgs = nn.Upsample(size=(show_size, show_size), mode='bilinear')(fake_imgs), nn.Upsample(size=(show_size, show_size), mode='bilinear')(imgs)
    fake_imgs.add_(1).div_(2).mul_(255), imgs.add_(1).div_(2).mul_(255)
    fake_imgs, imgs = fake_imgs.data.numpy(), imgs.data.numpy()
    fake_imgs, imgs = np.transpose(fake_imgs, (0, 2, 3, 1)), np.transpose(imgs, (0, 2, 3, 1))
    seg = nn.Upsample(size=(attn_size, attn_size), mode = 'nearest')(seg).cpu().numpy()
    text_convas = np.ones([nvis * FONT_MAX, (att_split_num + 2) * show_size, 3], dtype=np.uint8)
    for i in range(att_split_num + 2):
        istart = i * show_size
        iend = (i + 1) * show_size
        text_convas[:, istart:iend, :] = COLOR_DIC[i]
    text_map, sentence_list = drawCaption(nvis, text_convas, att, show_size, FONT_MAX)
    text_map = np.asarray(text_map).astype(np.uint8)
    merged = Image.new('RGBA', (show_size * (att_split_num + 2), (show_size * 2 + FONT_MAX) * nvis), (0, 0, 0, 0))
    img_height = 0
    for i in range(nvis):
        txt = text_map[i * FONT_MAX: (i + 1) * FONT_MAX]
        PIL_txt = Image.fromarray(np.uint8(txt))
        PIL_im, PIL_truth_im = Image.fromarray(np.uint8(fake_imgs[i])), Image.fromarray(np.uint8(imgs[i]))
        attn = attn_maps[i].view(1, len(vip_split_attr), attn_size * 2, attn_size)
        attn = nn.Upsample(size=(attn_size, attn_size), mode='nearest')(attn).cpu().detach()
        for j in range(cfg.GAN.P_NUM):
            nonzero = np.nonzero(seg[i,j,:,:])
            if nonzero[0].size > 0 and np.asarray(part2attr_dict[j+1]).size > 0:
                part_beforeNorm = attn[:, np.asarray(part2attr_dict[j+1]), :, :]
                attr_beforeNorm = part_beforeNorm[:, :, nonzero[0], nonzero[1]]
                minV = attr_beforeNorm.min()
                maxV = attr_beforeNorm.max()
                if minV == maxV:
                    attn[:, :, nonzero[0], nonzero[1]] = 1
                else:
                    # part_beforeNorm[:, :, nonzero[0], nonzero[1]] = (attr_beforeNorm - minV) / (maxV - minV)
                    # attn[:, np.asarray(part2attr_dict[j+1]), :, :] = part_beforeNorm
                    attn[:, :, nonzero[0], nonzero[1]] = (attn[:, :, nonzero[0], nonzero[1]] - minV) /(maxV - minV)
        attn = np.maximum(attn, 0)

        attn_max = attn.max(dim=1, keepdim=True)
        attn = torch.cat([attn_max[0], attn], 1)
        attn = attn.view(-1, 1, attn_size, attn_size)
        attn = attn.repeat(1, 3, 1, 1).data.numpy()
        attn = np.transpose(attn, (0, 2, 3, 1))

        num_attn = attn.shape[0]
        row_beforeNorm = []
        for j in range(num_attn):
            one_map = attn[j]
            if (show_size // attn_size) > 1:
                one_map = pyramid_expand(one_map, sigma=20, upscale=show_size // attn_size, multichannel=True)
            row_beforeNorm.append(one_map)
        merged.paste(PIL_txt, (0, img_height))
        merged.paste(PIL_truth_im, (0, img_height + FONT_MAX))
        img_left = show_size
        for j in range(att_split_num + 1):
            one_map = row_beforeNorm[j] * 255
            PIL_att = Image.fromarray(np.uint8(one_map))
            merged.paste(PIL_im, (img_left, img_height + FONT_MAX))
            merged.paste(PIL_att, (img_left, img_height + FONT_MAX + show_size))
            img_left += show_size
        img_height += show_size * 2 + FONT_MAX
    save_path = os.path.join(image_dir, str(epoch) + '.png')
    if raw:
        save_path = os.path.join(image_dir, str(epoch) + '_raw.png')
    merged.save(save_path)

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        if not hasattr(m.weight,'data'):
            return
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)

def trans_dict(netG_state_dict, state_dict):
    for name, para in netG_state_dict.items():
        if name not in state_dict:
            state_dict[name] = para
    return state_dict

def get_activations(images, model, batch_size, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                     must lie between 0 and 1.
    -- model       : Instance of inception model
    -- batch_size  : the images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size depends
                     on the hardware.
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    #d0 = images.shape[0]
    d0 = int(images.size(0))
    if batch_size > d0:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = d0

    n_batches = d0 // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, cfg.TEST.FID_DIMS))
    for i in range(n_batches):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches), end='', flush=True)
        start = i * batch_size
        end = start + batch_size

        '''batch = torch.from_numpy(images[start:end]).type(torch.FloatTensor)
        batch = Variable(batch, volatile=True)
        if cfg.CUDA:
            batch = batch.cuda()'''
        batch = images[start:end]

        pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)

    if verbose:
        print(' done')

    return pred_arr


def calculate_activation_statistics(act):
    """Calculation of the statistics used by the FID.
    Params:
    -- act      : Numpy array of dimension (n_images, dim (e.g. 2048)).
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def compute_inception_score(predictions, num_splits=1):
    # print('predictions', predictions.shape)
    scores = []
    for i in range(num_splits):
        istart = i * predictions.shape[0] // num_splits
        iend = (i + 1) * predictions.shape[0] // num_splits
        part = predictions[istart:iend, :]
        kl = part * \
            (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)


def negative_log_posterior_probability(predictions, num_splits=1):
    # print('predictions', predictions.shape)
    scores = []
    for i in range(num_splits):
        istart = i * predictions.shape[0] // num_splits
        iend = (i + 1) * predictions.shape[0] // num_splits
        part = predictions[istart:iend, :]
        result = -1. * np.log(np.max(part, 1))
        result = np.mean(result)
        scores.append(result)
    return np.mean(scores), np.std(scores)

def get_pt_version():
    raw_version = torch.__version__
    raw_version = raw_version.split('.')
    raw_version = [int(raw_version[i])*pow(10, len(raw_version)-i-1) for i in range(len(raw_version))]
    version = sum(raw_version)
    return version

class CKPT:
    def __init__(self, epoch, fid):
        self.epoch = epoch
        self.fid = fid

    def get_name(self):
        return 'ckpt_ep_{0:03d}_fid_{1:.2f}.pth'.format(self.epoch, self.fid)

    def get_epoch(self):
        return self.epoch

    def get_fid(self):
        return self.fid

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_fid(self, fid):
        self.fid = fid

def load_acts_data(split):
    filepath = os.path.join(cfg.DATA_DIR, '%s_acts.pickle'% split)
    if not os.path.isfile(filepath):
        print('Error: no such a file %s'%(filepath))
        return None
    else:
        with open(filepath, 'rb') as f:
            x = pickle.load(f)
            acts_dict = x[0]
            del x
            print('Load from: ', filepath)

    return acts_dict

################################### load img data #################################
def write_bytes(filenames, filepath, modality, postfix):
    # modality: 'images' / 'semantic_segmentations'
    # postfix: '.jpg' / 'npz'
    with open(filepath, 'wb') as wfid:
        for index in range(len(filenames)):
            if index % 500 == 0:
                print('%07d / %07d'%(index, len(filenames)))

            file_name = os.path.join(cfg.DATA_DIR, modality, '%s%s'%(filenames[index], postfix))
            with open(file_name, 'rb') as fid:
                fbytes = fid.read()

            wfid.write(struct.pack('i', len(fbytes)))
            wfid.write(fbytes)

def read_bytes(filenames, filepath):
    fbytes = []
    print('start loading bigfile (%0.02f GB) into memory' % (os.path.getsize(filepath)/1024/1024/1024))
    with open(filepath, 'rb') as fid:
        for index in range(len(filenames)):
            fbytes_len = struct.unpack('i', fid.read(4))[0]
            fbytes.append(fid.read(fbytes_len))

    return fbytes

def load_bytes_data(split, filenames, modality, postfix):
    filepath = os.path.join(cfg.DATA_DIR, split, '{}.bigfile'.format(modality))

    if not os.path.isfile(filepath):
        print('writing %s files'%(split))
        write_bytes(filenames, filepath, modality, postfix)

    fbytes = read_bytes(filenames, filepath)

    return fbytes