import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms.functional as F
from sklearn.preprocessing import StandardScaler
import pandas as pd
from PIL import Image
import yaml
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_photocast_dataloader(cfg):
    """
    
    """

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(cfg['img_size'], interpolation=torchvision.transforms.InterpolationMode.BICUBIC),  
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    datasets = []

    for camera in cfg["roundshot"]:      
        datasets.append(
           PhotocastDataset(os.path.join(cfg['dataset_path'], "v3", camera + "_train.csv"),
                              os.path.join(cfg['dataset_path'], "v3", camera + "_train_pairs.csv"),
                              "/mnt/photocast/data", #cfg['dataset_path'],
                              cfg['img_size']) 
        )
    dataset = torch.utils.data.ConcatDataset(datasets)
    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True)
    return dataloader


class PhotocastDataset(Dataset):
    """
    Photocastv3 dataset wrapper (panorama images and v2 transforms)
    """

    def __init__(self,
                 csv_file,
                 pairs_file,
                 data_root_dir,
                 img_size,
                 img_transform=None):
        """
        Args:
            csv_file:
            pairs_file:
            data_root_dir:
            img_size:
            img_transform:
        """
        self.data = pd.read_csv(csv_file, index_col=0)
        self.pairs = pd.read_csv(pairs_file, index_col=None, header=None)

        self.root_dir = data_root_dir

        # Scale weather data
        self.scaler = StandardScaler()
        self.weather = self.scaler.fit_transform(
            self.data.drop(columns=[
                'rowid',
                'identifier',
                'position',
                'station',
                'reference',
                'path',
                'mod',
                'x',
                'y',
                'pos',
                'cam',
                'latitude',
                'longitude',
                'north',
                'angle',
                'LP',
                'RP',
                'PN',
                'NT'], errors="ignore")
        )

        self.root_dir = data_root_dir
        self.img_size = img_size
        self.img_transform = img_transform

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        I0_idx, It_idx = self.pairs.iloc[index]
        I0 = Image.open(os.path.join(self.root_dir, self.data["path"][I0_idx]))
        It = Image.open(os.path.join(self.root_dir, self.data["path"][It_idx]))
        w0 = torch.Tensor(self.weather[I0_idx])
        wt = torch.Tensor(self.weather[It_idx])

        if self.img_transform is not None:
            I0, It = self.img_transform(I0, It)

        return I0, It, w0, wt