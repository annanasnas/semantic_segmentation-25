from torch.utils.data import Dataset
import glob
import os
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import albumentations as A


class CityScapes(Dataset):
    def __init__(self, root_dir, split='train', image_transform=None, image_size=(512, 1024)):
        super(CityScapes, self).__init__()

        self.root_dir = root_dir

        self.images_dir = os.path.join(root_dir, 'images', split)
        self.labels_dir = os.path.join(root_dir, 'gtFine', split)

        self.image_paths = sorted(glob.glob(os.path.join(self.images_dir, '**', '*_leftImg8bit.png'), recursive=True))
        self.label_paths = sorted(glob.glob(os.path.join(self.labels_dir, '**', '*_gtFine_labelTrainIds.png'), recursive=True))
        
        self.split = split
        self.image_transform = image_transform
        self.image_size = image_size

        assert len(self.image_paths) == len(self.label_paths), "Mismatch between image and label counts"

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path)

        image = transforms.Resize((self.image_size))(image)
        label = transforms.Resize((self.image_size), interpolation=Image.NEAREST)(label)
        label = torch.as_tensor(np.array(label), dtype=torch.long)

        # albumentations
        if isinstance(self.image_transform, A.Compose):
            img_np = np.array(image)
            transformed = self.image_transform(image=img_np)
            image = transformed["image"]
        else: # torch
            image = self.image_transform(image)

        return image, label

    def __len__(self):
        return len(self.image_paths)
