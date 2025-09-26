import os
import glob
import numpy as np
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

print("processing data...")

# directory setup
data_dir = '/home/omkos333/projects/brainseg/data/raw' # base data directory
save_dir = '/home/omkos333/projects/brainseg/data/processed' # new folder to save to

for sub_path in ("train/images", "train/masks", "test/images", "test/masks"):
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

# train/test split
N = len(flair_list)
train_size, test_size = 0.8, 0.2
train_idx, test_idx = train_test_split(np.arange(N), train_size=train_size, test_size=test_size, random_state=42)

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
    
    stack = stack[56:184, 56:184, 13:141]
    mask = mask[56:184, 56:184, 13:141]

    mask = to_one_hot(mask, num_classes=4)

    stack = stack[::2, ::2, ::2, ::] # shrink cropped images to (64, 64, 64, 3)
    mask = mask[::2, ::2, ::2, ::] # shrink cropped masks to (64, 64, 64, 4)

    stack = np.transpose(stack, (3, 0, 1, 2)) # channels first, (3, 64, 64, 64)
    mask = np.transpose(mask, (3, 0, 1, 2)) # (4, 64, 64, 64)
    
    train_idx = set(train_idx) # fast O(1) lookup with a set
    sub_path = "train" if image in train_idx else "test"

    image_dir = os.path.join(save_dir, sub_path, "images")
    mask_dir = os.path.join(save_dir, sub_path, "masks")

    np.save(os.path.join(image_dir, f"image_{image}.npy"), stack)
    np.save(os.path.join(mask_dir, f"mask_{image}.npy"), mask)

print("done")