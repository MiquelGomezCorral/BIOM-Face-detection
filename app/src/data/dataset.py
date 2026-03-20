import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from maikol_utils.print_utils import print_separator
from maikol_utils.file_utils import list_dir_files

from src.config import Configuration
from .filter import local_normalize_image

class FACES_DATASET(Dataset):
    def __init__(self, partition = "train", transform = None, CONFIG: Configuration = None):
        self.partition = partition
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.ToTensor(),
        ])
        self.config = CONFIG

        if self.partition == "train":
            self.data_paths, self.n = list_dir_files(self.config.train_f_path)
            # self.data_paths, self.n = list_dir_files(self.config.train_path)
        elif self.partition == "val":
            self.data_paths, self.n = list_dir_files(self.config.val_f_path)
            # self.data_paths, self.n = list_dir_files(self.config.val_path)
        else:
            self.data_paths, self.n = list_dir_files(self.config.test_f_path)
            # self.data_paths, self.n = list_dir_files(self.config.test_path)

        print(f" - Total data {self.partition}: {self.n} images")

    
    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.config.gray_scale:
            img = cv2.imread(self.data_paths[idx], cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(self.data_paths[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # # Filter for the images
        # img = local_normalize_image(self.config, img)

        # # # Rescale to [0, 255] preserving full dynamic range
        # img_min, img_max = img.min(), img.max()
        # if img_max > img_min:
        #     img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        # else:
        #     img = np.zeros_like(img, dtype=np.uint8)

        # Resize
        img = cv2.resize(img, (self.config.crop_size, self.config.crop_size), interpolation=cv2.INTER_AREA)
        
        # Convert to PIL for transforms (mode 'L' for grayscale, 'RGB' for colour)
        img = Image.fromarray(img)

        # data augmentation
        img_tensor = self.transform(img)

        # Label
        label = torch.tensor('person' in self.data_paths[idx], dtype=torch.long)
        return {"img": img_tensor, "label": label}
    


def load_faces(CONFIG: Configuration):
    print_separator(f"Loading FACES Dataset...")
    
    train_da = transforms.Compose([
        transforms.RandomHorizontalFlip(p=CONFIG.aug_prob),
        transforms.RandomApply([
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ], p=CONFIG.aug_prob),
        transforms.ToTensor(),
    ])
    train_da = transforms.Compose([
        transforms.RandomHorizontalFlip(p=CONFIG.aug_prob),
        # Spatial: Rotation + Translation + Scaling
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=15, 
                translate=(0.1, 0.1), 
                scale=(0.8, 1.2)
            )
        ], p=CONFIG.aug_prob),
        # Noise: Simulates sensor grain
        transforms.RandomApply([
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.02)
        ], p=CONFIG.aug_prob * 0.5), # Apply noise less frequently
        transforms.ToTensor(),
        # Standardize for the CNN
        transforms.Normalize(mean=[0.5], std=[0.5]) 
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    train_dataset = FACES_DATASET(partition="train", transform=train_da, CONFIG=CONFIG)
    val_dataset = FACES_DATASET(partition="val", transform=test_transform, CONFIG=CONFIG)
    test_dataset = FACES_DATASET(partition="test", transform=test_transform, CONFIG=CONFIG)

    # DataLoader Class
    train_dataloader = DataLoader(train_dataset, CONFIG.batch_size, shuffle=True,
                                  num_workers=CONFIG.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, CONFIG.batch_size, shuffle=False,
                                num_workers=CONFIG.num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, CONFIG.batch_size, shuffle=False,
                                 num_workers=CONFIG.num_workers, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader
