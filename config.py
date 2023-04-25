import torch

DATASET_URL = 'http://images.cocodataset.org/zips/train2014.zip'

TEST = True
LOAD = False
SAVE = True
SAVE_FREQ = 100
TEST_FREQ = 100
CHECKPOINT_FILE = 'checkpoint.tar'
LOSSES_FILE = 'losses.csv'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 1
BATCH_SIZE = 4
LEARNING_RATE = 0.001

STYLE_IMG = 'style_images/wave_crop.jpg'
CONTENT_WEIGHT = 1e0
STYLE_WEIGHT = 4e5
TV_WEIGHT = 0.0
