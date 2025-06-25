from torch.utils.data import Dataset
import glob
import os
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import albumentations as A


class GTA5Dataset(Dataset):
    def __init__(self, root_dir, image_transform=None, image_size=(720, 1280)):

        self.root_dir = root_dir

        self.images_dir = os.path.join(root_dir, 'images')
        self.labels_dir = os.path.join(root_dir, 'labels')

        self.image_paths = sorted(glob.glob(os.path.join(self.images_dir, '*.png')))
        self.label_paths = sorted(glob.glob(os.path.join(self.labels_dir, '*.png')))

        self.image_transform = image_transform
        self.image_size = image_size

        assert len(self.image_paths) == len(self.label_paths), "Mismatch between image and label counts"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path)

        image = transforms.Resize(self.image_size)(image)
        label = transforms.Resize(self.image_size, interpolation=Image.NEAREST)(label)

        label_np = np.array(label, dtype=np.uint8)
        label_remapped = self.remap_gta5_labels(label_np)
        label = torch.as_tensor(label_remapped, dtype=torch.long)

        # albumentations
        if isinstance(self.image_transform, A.Compose):
            transformed = self.image_transform(image=np.array(image), mask=label_remapped)
            image = transformed['image']
            label = transformed['mask'].long()
        else: # torch
            image = self.image_transform(image)

        return image, label
    

    def remap_gta5_labels(self, label_mask):

        # mapping dict
        GTA5_to_Cityscapes = {
            7: 0,    # Road
            8: 1,    # Sidewalk
            11: 2,   # Building
            12: 3,   # Wall
            13: 4,   # Fence
            17: 5,   # Pole
            19: 6,   # Traffic light
            20: 7,   # Traffic sign
            21: 8,   # Vegetation
            22: 9,   # Terrain
            23: 10,  # Sky
            24: 11,  # Person
            25: 12,  # Rider
            26: 13,  # Car
            27: 14,  # Truck
            28: 15,  # Bus
            31: 16,  # Train
            32: 17,  # Motorcycle
            33: 18,  # Bicycle
        }
        remapped = np.full_like(label_mask, fill_value=255)

        for gta_label, cityscapes_label in GTA5_to_Cityscapes.items():
            remapped[label_mask == gta_label] = cityscapes_label

        return remapped
        