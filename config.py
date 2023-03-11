# coding=utf-8
"""Configuration of an experiment."""
from easydict import EasyDict as edict

# Get the configuration dictionary whose keys can be accessed with dot.
CONFIG = edict()

# ******************************************************************************
# Experiment params
# ******************************************************************************

# self-supervised mode (SimClR-like methods compare two augmented views)
CONFIG.SSL = True
# the name of dataset dir
CONFIG.PATH_TO_DATASET = '/data/Users/pingan/micro_expression/crop2onsetBase/'
# Algorithm used for training: tcc, tcn, scl, classification.
CONFIG.TRAINING_ALGO = 'classification'

# ******************************************************************************
# Training params
# ******************************************************************************

CONFIG.TRAIN = edict()
# Number of training epoch.
CONFIG.TRAIN.MAX_EPOCHS = 500
# Number of samples in each batch.
CONFIG.TRAIN.BATCH_SIZE = 12
# Number of frames to use while training.
CONFIG.TRAIN.NUM_FRAMES = 240

# ******************************************************************************
# Eval params
# ******************************************************************************
CONFIG.EVAL = edict()
CONFIG.EVAL.CLASS_NUM = 2

# ******************************************************************************
# Model params
# ******************************************************************************
CONFIG.MODEL = edict()

CONFIG.MODEL.BASE_MODEL = edict()
CONFIG.MODEL.BASE_MODEL.OUT_CHANNEL = 2048
# The video will be sent to 2D resnet50 as batched frames,
# the max number of frame in each batch.
CONFIG.MODEL.BASE_MODEL.FRAMES_PER_BATCH = 40

# Select which layers to train.
# train_base defines how we want proceed with the base model.
# 'frozen' : Weights are fixed and batch_norm stats are also fixed.
# 'train_all': Everything is trained and batch norm stats are updated.
# 'only_bn': Only tune batch_norm variables and update batch norm stats.
CONFIG.MODEL.TRAIN_BASE = 'train_all'

# Paramters for transformers
CONFIG.MODEL.EMBEDDER_MODEL = edict()
CONFIG.MODEL.EMBEDDER_MODEL.HIDDEN_SIZE = 256
CONFIG.MODEL.EMBEDDER_MODEL.D_FF = 1024
CONFIG.MODEL.EMBEDDER_MODEL.NUM_HEADS = 8
CONFIG.MODEL.EMBEDDER_MODEL.NUM_LAYERS = 3
CONFIG.MODEL.EMBEDDER_MODEL.FC_LAYERS = [(256, True), (256, True)]
CONFIG.MODEL.EMBEDDER_MODEL.CAPACITY_SCALAR = 2
CONFIG.MODEL.EMBEDDER_MODEL.EMBEDDING_SIZE = 128
CONFIG.MODEL.EMBEDDER_MODEL.FC_DROPOUT_RATE = 0.1

# The projection head introduced by SimCLR
CONFIG.MODEL.PROJECTION = True
CONFIG.MODEL.PROJECTION_HIDDEN_SIZE = 512
CONFIG.MODEL.PROJECTION_SIZE = 128

# ******************************************************************************
# our Sequential Contrastive Loss params
# ******************************************************************************
# Read our CARL paper for better understanding
CONFIG.SCL = edict()
CONFIG.SCL.LABEL_VARIENCE = 10.0
CONFIG.SCL.SOFTMAX_TEMPERATURE = 0.1
CONFIG.SCL.POSITIVE_TYPE = 'gauss'
CONFIG.SCL.NEGATIVE_TYPE = 'single_noself'
CONFIG.SCL.POSITIVE_WINDOW = 5

# ******************************************************************************
# TCC params
# ******************************************************************************
CONFIG.TCC = edict()
CONFIG.TCC.CYCLE_LENGTH = 2
CONFIG.TCC.LABEL_SMOOTHING = 0.1
CONFIG.TCC.SOFTMAX_TEMPERATURE = 0.1
CONFIG.TCC.LOSS_TYPE = 'regression_mse_var'
CONFIG.TCC.NORMALIZE_INDICES = True
CONFIG.TCC.VARIANCE_LAMBDA = 0.001
CONFIG.TCC.FRACTION = 1.0
CONFIG.TCC.HUBER_DELTA = 0.1
CONFIG.TCC.SIMILARITY_TYPE = 'l2'  # l2, cosine

# ******************************************************************************
# Time Contrastive Network params
# ******************************************************************************
CONFIG.TCN = edict()
CONFIG.TCN.POSITIVE_WINDOW = 5
CONFIG.TCN.REG_LAMBDA = 0.002

# ******************************************************************************
# Optimizer params
# ******************************************************************************
CONFIG.OPTIMIZER = edict()
# Supported optimizers are: AdamOptimizer, MomentumOptimizer
CONFIG.OPTIMIZER.TYPE = 'AdamOptimizer'
CONFIG.OPTIMIZER.WEIGHT_DECAY = 0.00001
CONFIG.OPTIMIZER.GRAD_CLIP = 10

CONFIG.OPTIMIZER.LR = edict()
# Initial learning rate for optimizer.
CONFIG.OPTIMIZER.LR.INITIAL_LR = 0.0001
# Learning rate decay strategy.
# Currently Supported strategies: fixed, cosine, cosinewarmup
CONFIG.OPTIMIZER.LR.DECAY_TYPE = 'cosine'
CONFIG.OPTIMIZER.LR.WARMUP_LR = 0.0001
CONFIG.OPTIMIZER.LR.FINAL_LR = 0.0
CONFIG.OPTIMIZER.LR.NUM_WARMUP_STEPS = 1


def get_cfg():
    return CONFIG
