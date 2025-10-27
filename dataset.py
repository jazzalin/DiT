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
           Photocastv2Dataset(os.path.join(cfg['dataset_path'], "v3", camera + "_train.csv"),
                              os.path.join(cfg['dataset_path'], "v3", camera + "_train_pairs.csv"),
                              "/mnt/photocast/data", #cfg['dataset_path'],
                              cfg['img_size']) 
        )
    dataset = torch.utils.data.ConcatDataset(datasets)
    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True)
    return dataloader


class Photocastv2Dataset(Dataset):
    """
    Photocastv2 dataset wrapper
    """

    def __init__(self,
                 csv_file,
                 pairs_file,
                 data_root_dir,
                 img_size):
        """
        Args:
            csv_file:
            pairs_file:
            data_root_dir:
            img_size:
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
                'LP'], errors="ignore")
        )

        self.root_dir = data_root_dir
        self.img_size = img_size

    def transform(self, images, crop=None, aspect=1.5):
        # Panorama random crop: apply the same crop to all images in
        # {I0, It_1, ..., It}
        # if crop is None:
        #     i, j, h, w = PanoramaCrop.get_params(images[0], aspect)
        # else:
        #     i, j, h, w = crop
        
        tf_images = []
        for img in images:
            # img = F.crop(img, i, j, h, w)
            img = F.resize(img, self.img_size[0], interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
            img = F.center_crop(img, self.img_size)
            img = F.to_tensor(img)
            img = F.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            tf_images.append(img)
        
        return tf_images


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

        tf_img = self.transform([I0, It])
        
        return tf_img[0], tf_img[1], w0, wt
    
    def get_crop(self, index, crop):
        if torch.is_tensor(index):
            index = index.tolist()

        I0_idx, It_idx = self.pairs.iloc[index]
        I0 = Image.open(os.path.join(self.root_dir, self.data["path"][I0_idx]))
        It = Image.open(os.path.join(self.root_dir, self.data["path"][It_idx]))
        w0 = torch.Tensor(self.weather[I0_idx])
        wt = torch.Tensor(self.weather[It_idx])

        tf_img, crop = self.transform([I0, It], crop=crop)
        
        return tf_img[0], tf_img[1], w0, wt, crop
    
    def get_pairs(self, reference, lead=None):
        """
        Get (I0, It) pairs based on reference and lead time (optional)
        """

        assert pd.tseries.api.guess_datetime_format(reference) == "%Y-%m-%d %H:%M:%S", \
            "Ensure reference is in '%Y-%m-%d %H:%M:%S' format"
        

        data_idx = self.data.query("reference == @reference").index

        pairs_idx = pd.DataFrame()
        if len(data_idx) == 1:
            pairs_idx = self.pairs[self.pairs[0] == data_idx[0]]

            if lead is not None:
                try:
                    pairs_idx = pairs_idx.iloc[:lead]
                except IndexError as err:
                    print(err)
        return pairs_idx
    
    def get_image(self, index, transforms=None):
        if torch.is_tensor(index):
            index = index.tolist()

        I0_idx, It_idx = self.pairs.iloc[index]
        I0 = Image.open(os.path.join(self.root_dir, self.data["path"][I0_idx]))
        W, H  = I0.size
        It = Image.open(os.path.join(self.root_dir, self.data["path"][It_idx]))
        w0 = torch.Tensor(self.weather[I0_idx])
        wt = torch.Tensor(self.weather[It_idx])

        if transforms is not None:
            It = transforms(It)
            I0 = transforms(I0)
        
        return I0, It, w0, wt