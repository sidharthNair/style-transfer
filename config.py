import torch

LOAD = True
SAVE = True
FILE = 'checkpoint.tar'
LOSSES_FILE = 'losses.csv'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WORKERS = 4
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
