import numpy as np
import os
import imageio
import torch

from PIL import Image
from torchvision.utils import save_image
from config import *
from model import Net
from data import load_img, GENERAL_TRANSFORM, INV_NORMALIZE

def test_net(net, prefix='', folder='test_images'):
    files = os.listdir(folder)
    if not os.path.exists(f'{folder}_output/'):
        os.makedirs(f'{folder}_output/')

    net.eval()
    for file in files:
        content_img = load_img(f'{folder}/' + file)
        content_img = GENERAL_TRANSFORM(content_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            stylized = net(content_img)
            save_image(INV_NORMALIZE(stylized), f'{folder}_output/{prefix}stylized_' + file)
    net.train()

def create_gif(contains='', folder='test_images_output', skip=0, start=0, num_frames=None):
    files = os.listdir(folder)
    files = [file for file in files if contains in file]
    sorted_files = sorted(files, key = lambda x: int(x.split("_")[0]))
    imgs = [np.array(Image.open(os.path.join(folder, sorted_files[i]))) for i in range(start, min(len(sorted_files), start + num_frames * (skip + 1)), skip + 1)]
    imageio.mimsave(f'{folder}/generated{contains}.gif', imgs, duration=(5/num_frames))

def main():
    net = Net().to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
    net.load_state_dict(checkpoint["state_dict"])
    test_net(net)

if __name__ == "__main__":
    main()