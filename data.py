import os
import zipfile
import numpy as np
import PIL.Image as Image
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import albumentations as A
import albumentations.pytorch as AT
from torch.utils.data import Dataset, DataLoader
from torch.hub import download_url_to_file

DATASET_URL = 'http://images.cocodataset.org/zips/train2014.zip'

# Define the transform to normalize the data
transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    AT.ToTensorV2(),
])

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
        img = np.array(Image.open(os.path.join(dir, file)))
        img = transform(image=img)["image"]
        return img

if __name__ == '__main__':
    cd = CustomDataset('dataset')
    print(cd.__getitem__(0).shape)