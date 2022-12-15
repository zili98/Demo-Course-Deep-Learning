from yacs.config import CfgNode as CN

_C = CN()

_C.DATASET = CN()
# Path to directory containing the train, validation and test dataset.
_C.DATASET.DATA_DIR = ''
# Path to input mfcc features.
_C.DATASET.TRAIN_FILE = ''
# Path to labels.
_C.DATASET.VAL_FILE = ''
# Path to original labels.
_C.DATASET.TEST_FILE = ''

_C.DATALOADER = CN()
# Batch size.
_C.DATALOADER.BATCH_SIZE= 12
# Number of subprocesses to use for data loading.
_C.DATALOADER.NUM_WORKERS = 4
_C.DATALOADER.CROSS_VALIDATE = True
_C.DATALOADER.K_FOLD = 5

_C.MODEL = CN()
# Model name.
_C.MODEL.NAME = ''
# Number of ground truth classes.
_C.MODEL.NUM_CLASSES = 23
_C.MODEL.LOSS_FUNC = ''
# List of loss functions to use. Each list item should have a NAME and ARGS
# which is a dictionary of arguments to pass to the loss function class.

_C.LOSSES = []

_C.TRAIN_ARGS = CN()
# Base learning rate.
_C.TRAIN_ARGS.BASE_LR = 0.001
# Weight decay in optimizer.
_C.TRAIN_ARGS.WEIGHT_DECAY = 0.001
# Number of epochs to train for.
_C.TRAIN_ARGS.NUM_EPOCHS = 30
# The factor to reduce the current learning rate by.
_C.TRAIN_ARGS.LR_SCHEDULER_FACTOR = 0.1
# Number of epochs with no improvement after which learning rate will be reduced.
_C.TRAIN_ARGS.LR_SCHEDULER_PATIENCE = 2
# Minimum learning rate.
_C.TRAIN_ARGS.MIN_LR = 1e-6
# Number of validation checks with no improvement after which training will be stopped. 0 = no early stopping.
_C.TRAIN_ARGS.EARLY_STOPPING_PATIENCE = 0
# Name of LR scheduler to use.
_C.TRAIN_ARGS.LR_SCHEDULER = 'CosineAnnealingLR'
_C.TRAIN_ARGS.OPTIMIZER = 'SGD'
# Number of epochs for the first restart.
_C.TRAIN_ARGS.WARM_RESTART_EPOCH = 20
_C.TRAIN_ARGS.WARM_UP_EPOCH = 10
_C.TRAIN_ARGS.MAX_EPOCHS = 10
_C.TRAIN_ARGS.CYCLICAL_EPOCHS = 100

# Name of the run to log in mlflow.
_C.RUN_NAME = ''
# Device to run training on.
_C.DEVICE = 'cuda'
# Number of GPUs to use for training.
_C.NUM_GPUS = 4
# Keep only the top k checkpoints. k = -1 to keep all checkpoints.
_C.SAVE_TOP_K = 3
# Save frequency in number of epochs.
_C.SAVE_FREQ = 1
# Random seed/
_C.SEED = 42


def get_cfg_defaults() -> CN:
    """Gets a yacs CfgNode object with default values for an experiment."""
    return _C.clone()


def get_cfg_from_yaml(yaml_path: str) -> CN:
    """Gets a yacs CfgNode object with default values and overwrites it with the values from the yaml file.
    
    Args:
        yaml_path: Path to yaml file containing the custom configuration.

    Returns:
        Merged config.
    """
    cfg = get_cfg_defaults()
    cfg.merge_from_file(yaml_path)

    return cfg
