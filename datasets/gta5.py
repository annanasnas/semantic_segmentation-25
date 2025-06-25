from torch.utils.data import Dataset
import glob
import os
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import albumentations as A
from torch.fft import fft2, ifft2
import random


class GTA5Dataset(Dataset):
    def __init__(self, root_dir, image_transform=None, image_size=(720, 1280), FDA=False, beta=0.01):

        self.root_dir = root_dir
        self.cityscapes_image_paths = sorted(glob.glob(os.path.join(os.path.join('/content/semantic_segmentation-25/datasets/data/Cityscapes/Cityspaces', 'images', 'train'), '**', '*_leftImg8bit.png'), recursive=True))

        self.images_dir = os.path.join(root_dir, 'images')
        self.labels_dir = os.path.join(root_dir, 'labels')

        self.image_paths = sorted(glob.glob(os.path.join(self.images_dir, '*.png')))
        self.label_paths = sorted(glob.glob(os.path.join(self.labels_dir, '*.png')))

        self.image_transform = image_transform
        self.image_target_transform = transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
                                                          ])
        self.image_size = image_size

        self.FDA = FDA
        self.beta = beta
        self.to_tensor = transforms.ToTensor()

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

        if self.FDA:
            trg_path = random.choice(self.cityscapes_image_paths)
            trg_image = Image.open(trg_path).convert("RGB")
            trg_image = transforms.Resize(self.image_size)(trg_image)

            src_tensor = self.to_tensor(image).unsqueeze(0)
            trg_tensor = self.to_tensor(trg_image).unsqueeze(0)

            with torch.no_grad():
                fda_img_tensor = self.FDA_source_to_target(src_tensor, trg_tensor, beta=self.beta)[0]
            
            fda_np = fda_img_tensor.permute(1, 2, 0).cpu().numpy()
            fda_np = (fda_np * 255).clip(0, 255).astype(np.uint8)

            transformed = self.image_transform(image=fda_np, mask=label_remapped)
            image = transformed['image']
            label = transformed['mask'].long()

            trg_image = self.image_target_transform(trg_image)

            return image, label, trg_image

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


    def low_freq_mutate(self, amp_s, amp_t, beta):

        a_s = torch.fft.fftshift(amp_s, dim=(-2,-1))
        a_t = torch.fft.fftshift(amp_t, dim=(-2,-1))

        B,C,H,W = a_s.shape
        b_h = int(H * beta/2)
        b_w = int(W * beta/2)
        h1, h2 = H//2 - b_h, H//2 + b_h
        w1, w2 = W//2 - b_w, W//2 + b_w

        a_s[:, :, h1:h2, w1:w2] = a_t[:, :, h1:h2, w1:w2]

        return torch.fft.ifftshift(a_s, dim=(-2,-1))


    def FDA_source_to_target(self, x_s, x_t, beta):
        # 1. forward 2D FFT
        F_s = fft2(x_s, dim=(-2, -1))
        F_t = fft2(x_t, dim=(-2, -1))
        # 2. split amplitude & phase
        A_s, P_s = torch.abs(F_s), torch.angle(F_s)
        A_t = torch.abs(F_t)
        # 3. swap low-frequencies
        A_mix = self.low_freq_mutate(A_s.clone(), A_t.clone(), beta)
        # 4. reassemble complex spectrum
        F_mix = A_mix * torch.exp(1j * P_s)
        # 5. inverse FFT back to image
        x_s_to_t = ifft2(F_mix, dim=(-2, -1)).real
        return x_s_to_t
        