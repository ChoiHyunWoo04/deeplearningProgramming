import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = [""]

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 256 # default: 128
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ""
# Dataset name
_C.DATA.DATASET = "cifar"
# Input image size
_C.DATA.IMG_SIZE = 32
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = "bicubic"
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = "part"
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 0 # default: 8

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = "hrnet"
# Model name
_C.MODEL.NAME = "hrnet_w18"
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ""
_C.MODEL.RESUME_ONLY_MODEL = False
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 100
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

# HRNet parameters / .yaml 파일의 파라미터들 중 설정되지 않은 파라미터들에 default 값을 주는 역할
_C.MODEL.HRNET = CN()
_C.MODEL.HRNET.DROP_PATH_RATE = 0.2

_C.MODEL.HRNET.STAGE1 = CN()
_C.MODEL.HRNET.STAGE1.NUM_MODULES = 1
_C.MODEL.HRNET.STAGE1.NUM_BRANCHES = 1
_C.MODEL.HRNET.STAGE1.NUM_BLOCKS = [2]
_C.MODEL.HRNET.STAGE1.NUM_CHANNELS = [64]
_C.MODEL.HRNET.STAGE1.BLOCK = "BOTTLENECK"

_C.MODEL.HRNET.STAGE2 = CN()
_C.MODEL.HRNET.STAGE2.NUM_MODULES = 1
_C.MODEL.HRNET.STAGE2.NUM_BRANCHES = 2
_C.MODEL.HRNET.STAGE2.NUM_BLOCKS = [2, 2]
_C.MODEL.HRNET.STAGE2.NUM_CHANNELS = [18, 36]
_C.MODEL.HRNET.STAGE2.BLOCK = "BASIC"

_C.MODEL.HRNET.STAGE3 = CN()
_C.MODEL.HRNET.STAGE3.NUM_MODULES = 3
_C.MODEL.HRNET.STAGE3.NUM_BRANCHES = 3
_C.MODEL.HRNET.STAGE3.NUM_BLOCKS = [2, 2, 2]
_C.MODEL.HRNET.STAGE3.NUM_CHANNELS = [18, 36, 72]
_C.MODEL.HRNET.STAGE3.BLOCK = "BASIC"

_C.MODEL.HRNET.STAGE4 = CN()
_C.MODEL.HRNET.STAGE4.NUM_MODULES = 2
_C.MODEL.HRNET.STAGE4.NUM_BRANCHES = 4
_C.MODEL.HRNET.STAGE4.NUM_BLOCKS = [2, 2, 2, 2]
_C.MODEL.HRNET.STAGE4.NUM_CHANNELS = [18, 36, 72, 144]
_C.MODEL.HRNET.STAGE4.BLOCK = "BASIC"

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 200 # default: 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 0
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = "cosine"
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = "adamw"
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Mixed precision opt level, if O0, no amp is used ('O0', 'O1', 'O2')
# overwritten by command line argument
_C.AMP_OPT_LEVEL = "O1"
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ""
# Tag of experiment, overwritten by command line argument
_C.TAG = "default"
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, "r") as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault("BASE", [""]):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print("=> merge config from {}".format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.zip:
        config.DATA.ZIP_MODE = True
    if args.cache_mode:
        config.DATA.CACHE_MODE = args.cache_mode
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    # if args.amp_opt_level:
    #     config.AMP_OPT_LEVEL = args.amp_opt_level
    if args.output:
        config.OUTPUT = args.output
    if args.tag:
        config.TAG = args.tag
    if args.eval:
        config.EVAL_MODE = True
    if args.throughput:
        config.THROUGHPUT_MODE = True

    # set local rank for distributed training
    config.LOCAL_RANK = args.local_rank

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config


def get_config_default(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()

    return config