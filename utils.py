import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from monai.transforms import Compose, RandFlipd, RandRotate90d, RandZoomd, RandScaleIntensityd, RandGaussianNoised

class npy_dataset(Dataset):
    def __init__(self, image_dir, mask_dir, data_augmentation=False):
        self.image_paths = sorted([image_dir + '/' + f for f in os.listdir(image_dir)])
        self.mask_paths = sorted([mask_dir + '/' + f for f in os.listdir(mask_dir)])

        self.data_augmentation = data_augmentation
        if self.data_augmentation:
            self.transforms = Compose(
                [
                    RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0), # vertical axis flips
                    RandRotate90d(keys=["image", "mask"], prob=0.3, spatial_axes=(1, 2)), # 90 degree rotations
                    RandZoomd(keys=["image", "mask"], prob=0.2, min_zoom=0.9, max_zoom=1.1, mode=("trilinear", "nearest")), # small zooms
                    RandScaleIntensityd(keys=["image"], prob=0.3, factors=0.1), # intensity scaling
                    RandGaussianNoised(keys=["image"], prob=0.2, std=0.01), # some noise
                ]
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.load(self.image_paths[idx]) # (3, 64, 64, 64)
        mask = np.load(self.mask_paths[idx]) # (4, 64, 64, 64)

        if self.data_augmentation:
            sample = self.transforms({"image": image, "mask": mask})
            image, mask = sample["image"], sample["mask"] # already tensors
            return image, mask

        # convert to tensors
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        return image, mask

def compute_class_weights(data_loader, num_classes, include_background, device):
    all_counts = torch.zeros(num_classes, dtype=torch.float)
    
    for _, mask in data_loader:
        class_indices = torch.argmax(mask, dim=1) # converts one-hot to class indices, (b, num_classes, d, h, w) to (b, d, h, w)
        counts = torch.bincount(class_indices.flatten(), minlength=num_classes)
        all_counts += counts.float()
    
    class_counts = all_counts if include_background else all_counts[1:]
    class_counts = torch.clamp(class_counts, min=1.0) # prevent division by 0 in case a class has 0 pixels

    class_weights = class_counts.sum() / (len(class_counts) * class_counts)
    
    return class_weights.to(device)

def one_hot_from_logits(logits, num_classes):
    labels = torch.argmax(logits, dim=1)
    one_hot = F.one_hot(labels, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    return one_hot