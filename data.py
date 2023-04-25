import os
import zipfile
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset
from torch.hub import download_url_to_file
from config import DATASET_URL

DATASET_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

GENERAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

INV_NORMALIZE = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
)

def load_img(path):
    img = Image.open(path).convert('RGB')
    return img

class CustomDataset(Dataset):
    def __init__(self, path):
        super().__init__()

        if not os.path.exists(path):
            os.makedirs(path)
            tmp = os.path.join(path, 'dataset.zip')
            print(f'Downloading dataset from {DATASET_URL} to {tmp}...')
            download_url_to_file(DATASET_URL, tmp)

            print(f'Unzipping {tmp}...')
            with zipfile.ZipFile(tmp) as zf:
                zf.extractall(path=path)
            os.remove(tmp)
        else:
            print(f'Dataset path exists, skipping download')

        self.data = []
        self.root = path
        self.classes = os.listdir(path)
        for i in range(len(self.classes)):
            files = os.listdir(os.path.join(path, self.classes[i]))
            self.data += list(zip(files, [i] * len(files)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file, label = self.data[index]
        dir = os.path.join(self.root, self.classes[label])
        return DATASET_TRANSFORM(load_img(os.path.join(dir, file)))

if __name__ == '__main__':
    cd = CustomDataset('dataset')
    img = cd[0]
    print(img.shape)
    plt.imshow(np.moveaxis(img.cpu().detach().numpy(), 0, -1))
    plt.show()