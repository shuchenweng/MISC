import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.DATASET_NAME = ''
__C.CONFIG_NAME = ''
__C.DATA_DIR = ''
__C.MODEL_DIR = ''
__C.PRETRAINED_DIR = ''
__C.CKPT = ''
__C.PATH_SEPARATOR = '\\'

__C.WORKERS = 0
__C.GPU_IDS = []

__C.MODEL = edict()

__C.TREE = edict()
__C.TREE.BRANCH_NUM = 3
__C.TREE.BASE_SIZE_WIDTH = 64
__C.TREE.BASE_SIZE_HEIGHT = 128

__C.IMAGE = edict()
__C.IMAGE.EMBEDDING_DIM = 256

__C.TEXT = edict()
__C.TEXT.EMBEDDING_DIM = 256
__C.TEXT.SENTENCE_DIM = 256*11

__C.TRAIN = edict()
__C.TRAIN.FLAG = True
__C.TRAIN.USE_MLT = False
__C.TRAIN.CKPT_UPLIMIT = 3
__C.TRAIN.NET_LE = ''
__C.TRAIN.NET_IE = ''
__C.TRAIN.UNI_BATCH_SIZE = 8
__C.TRAIN.BATCH_SIZE = 8
__C.TRAIN.LABEL_EMB_NUM = 512
__C.TRAIN.ATT_EMB_NUM = 512
__C.TRAIN.SNAPSHOT_INTERVAL = 10
__C.TRAIN.PRINT_INTERVAL = 100
__C.TRAIN.MAX_EPOCH = 600
__C.TRAIN.DISCRIMINATOR_LR = 0.0002
__C.TRAIN.GENERATOR_LR = 0.0002

__C.TRAIN.SMOOTH = edict()
__C.TRAIN.SMOOTH.BETA = 1.0
__C.TRAIN.SMOOTH.GAMMA1 = 5.0
__C.TRAIN.SMOOTH.GAMMA2 = 10.0
__C.TRAIN.SMOOTH.GAMMA3 = 5.0
__C.TRAIN.SMOOTH.LAMBDA = 20.0
__C.TRAIN.SMOOTH.LAMBDA2 = 4.0
__C.TRAIN.SMOOTH.LAMBDA3 = 0.03
__C.TRAIN.SMOOTH.LAMBDA4 = 1.0

__C.GAN = edict()
__C.GAN.DF_DIM = 64
__C.GAN.GF_DIM = 32
__C.GAN.Z_DIM = 100
__C.GAN.CONDITION_DIM = 100
__C.GAN.P_NUM = 19
__C.GAN.Z_MLP_NUM = 4

__C.TEST = edict()
__C.TEST.UNI_BATCH_SIZE = 8
__C.TEST.BATCH_SIZE = 8
__C.TEST.FID_DIMS = 2048 # Dimensionality of features returned by Inception
__C.TEST.PRINT_INTERVAL = 100

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return


    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)
