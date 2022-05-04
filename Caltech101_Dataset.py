from torch.utils.data import Dataset
import pandas as pd
from PIL import Image


class Caltech101Dataset(Dataset):
    def __init__(self, csv_dir, transform=None):
        """
        Make a dataset from the csv.
        :param csv_dir: directory of csv of the img(train/valid/test) fold.
        :param transform: transform for img.
        """
        self.csv_info = pd.read_csv(csv_dir).values
        self.transform = transform

    def __getitem__(self, index):
        img_dir, label = self.csv_info[index, 0], self.csv_info[index, 1]
        img = Image.open(img_dir).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.csv_info)

