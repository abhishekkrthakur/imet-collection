import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import os

device = torch.device("cuda:0")
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CollectionsDataset(Dataset):

    def __init__(self, csv_file, root_dir, num_classes, image_size, folds=None, transform=None):
        if folds is None:
            folds = []
        self.data = pd.read_csv(csv_file)
        if len(folds) > 0:
            self.data = self.data[self.data.fold.isin(folds)].reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.num_classes = num_classes
        self.image_size = image_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.loc[idx, 'id'] + '.png')
        image = Image.open(img_name)
        labels = self.data.loc[idx, 'attribute_ids']
        labels = labels.split()

        label_tensor = torch.zeros(self.num_classes)
        for i in labels:
            label_tensor[int(i)] = 1

        if self.transform:
            image = self.transform(image)


        return {'image': image,
                'labels': label_tensor
                }


class CollectionsDatasetTest(Dataset):

    def __init__(self, csv_file, root_dir, image_size, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.loc[idx, 'id'] + '.png')
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)

        return {'image': image}
