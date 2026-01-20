# data hyperparameters
TRAIN_SIZE = 0.8
TEST_SIZE = 0.2
USE_DATA_AUG = True
BATCH_SIZE = 2
SHUFFLE = True

# model hyperparameters
IN_CH = 3
OUT_CH = 4
NUM_FILTERS = 64
NUM_HEADS = 4
USE_DROPOUT = True
DROPOUT_PROB = 0.2

# optimizer hyperparameters
LR = 1e-4
BETAS = (0.9, 0.999)
WEIGHT_DECAY = 1e-4

# scheduler hyperparameters
SCHED_MIN_LR = 1e-6

# loss hyperparameters
USE_BACKGROUND = False
GAMMA = 1.0
L_DICE = 1.0
L_FOCAL = 1.0

# training hyperparameters
NUM_EPOCHS = 400
LOG_EVERY = 25