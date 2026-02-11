import os
import glob
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from hyperparameters import TRAIN_SIZE, VAL_SIZE, TEST_SIZE

print("processing data...")

# directory setup
data_dir = '/home/omkos333/projects/brainseg/data/raw' # base data directory
save_dir = '/home/omkos333/projects/brainseg/data/processed' # new folder to save to

for sub_path in ("train/images", "train/masks", "val/images", "val/masks", "test/images", "test/masks"):
    os.makedirs(os.path.join(save_dir, sub_path), exist_ok=True)

# creating a list of all file paths for each MRI scan type and the corresponding masks
flair_list = sorted(os.path.normpath(i) for i in glob.glob(data_dir + '/BraTS20_Training_*/BraTS20_Training_*_flair.nii'))
t1ce_list = sorted(os.path.normpath(i) for i in glob.glob(data_dir + '/BraTS20_Training_*/BraTS20_Training_*_t1ce.nii'))
t2_list = sorted(os.path.normpath(i) for i in glob.glob(data_dir + '/BraTS20_Training_*/BraTS20_Training_*_t2.nii'))
mask_list = sorted(os.path.normpath(i) for i in glob.glob(data_dir + '/BraTS20_Training_*/BraTS20_Training_*_seg.nii'))

# convert a (d, h, w) integer mask to one-hot (d, h, w, num_classes)
def to_one_hot(mask, num_classes):
    shape = mask.shape
    one_hot = np.zeros(shape + (num_classes,), dtype=np.uint8)
    for c in range(num_classes):
        one_hot[..., c] = (mask == c).astype(np.uint8)
    return one_hot

# train/validation/test split
N = len(flair_list)
train_size, val_size, test_size = TRAIN_SIZE, VAL_SIZE, TEST_SIZE
train_idx, test_idx = train_test_split(np.arange(N), train_size=train_size + val_size, test_size=test_size, random_state=42)
train_idx, val_idx = train_test_split(train_idx, test_size=val_size / (train_size + val_size), random_state=42)

# fast O(1) lookup with a set
train_idx = set(train_idx)
val_idx = set(val_idx)

# data processing loop
scaler = MinMaxScaler() # used to scale all voxels to [0, 1]

for image in range(N): # the length of all scan-specific filepath lists is the same
    flair = nib.load(flair_list[image]).get_fdata()
    flair = scaler.fit_transform(flair.reshape(-1, flair.shape[-1])).reshape(flair.shape)

    t1ce = nib.load(t1ce_list[image]).get_fdata()
    t1ce = scaler.fit_transform(t1ce.reshape(-1, t1ce.shape[-1])).reshape(t1ce.shape)

    t2 = nib.load(t2_list[image]).get_fdata()
    t2 = scaler.fit_transform(t2.reshape(-1, t2.shape[-1])).reshape(t2.shape)

    mask = nib.load(mask_list[image]).get_fdata()
    mask = mask.astype(np.uint8)
    mask[mask == 4] = 3 # reassign label 4 to label 3 since label 3 is missing

    stack = np.stack([flair, t1ce, t2], axis=3)
    
    # hardcoded crop since all volumes are (240, 240, 155)
    stack = stack[56:184, 56:184, 13:141]
    mask = mask[56:184, 56:184, 13:141]

    stack = zoom(stack, (0.5, 0.5, 0.5, 1), order=1) # shrink cropped images to (64, 64, 64, 3)
    mask = zoom(mask, (0.5, 0.5, 0.5), order=0) # shrink cropped masks to (64, 64, 64)

    mask = to_one_hot(mask, num_classes=4)

    stack = np.transpose(stack, (3, 0, 1, 2)) # channels first, (3, 64, 64, 64)
    mask = np.transpose(mask, (3, 0, 1, 2)) # (4, 64, 64, 64)
    
    if image in train_idx:
        sub_path = "train"
    elif image in val_idx:
        sub_path = "val"
    else:
        sub_path = "test"

    image_dir = os.path.join(save_dir, sub_path, "images")
    mask_dir = os.path.join(save_dir, sub_path, "masks")

    np.save(os.path.join(image_dir, f"image_{image}.npy"), stack)
    np.save(os.path.join(mask_dir, f"mask_{image}.npy"), mask)

print("done")