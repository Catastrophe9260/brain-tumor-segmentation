import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from monai.metrics import DiceMetric, HausdorffDistanceMetric

from model import ResAtt3DUNet
from utils import npy_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def one_hot_from_logits(logits):
    labels = torch.argmax(logits, dim=1) # [B, D, H, W]
    one_hot = F.one_hot(labels, num_classes=4).permute(0, 4, 1, 2, 3).float()
    return one_hot

def validate(model, test_data):
    model.eval()

    dice_mean_metric = DiceMetric(include_background=False, reduction="mean")
    dice_per_class_metric = DiceMetric(include_background=False, reduction="mean_channel")
    hd95_mean_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")

    with torch.no_grad():
        for img, true_mask in test_data:
            img, true_mask = img.to(device), true_mask.to(device)

            logits = model(img)
            pred_mask = one_hot_from_logits(logits)

            dice_mean_metric(pred_mask, true_mask)
            dice_per_class_metric(pred_mask, true_mask)
            hd95_mean_metric(pred_mask, true_mask)
    
    mean_dice = dice_mean_metric.aggregate().item()
    per_class_dice = dice_per_class_metric.aggregate().flatten().tolist()
    mean_hd95 = hd95_mean_metric.aggregate().item()

    dice_mean_metric.reset()
    dice_per_class_metric.reset()
    hd95_mean_metric.reset()

    return {"mean_dice": mean_dice, "per_class_dice": per_class_dice, "mean_hd95": mean_hd95}

def main():

    print("starting test loop...")

    # data setup
    data_dir = '/home/omkos333/projects/brain_tumor_seg/data/processed'

    test_ds = npy_dataset(
        os.path.join(data_dir, 'test', 'images'),
        os.path.join(data_dir, 'test', 'masks'),
        data_augmentation=False
    )

    test_loader = DataLoader(test_ds, batch_size=2)

    # model setup
    model = ResAtt3DUNet(num_filters=64).to(device)
    model.load_state_dict(torch.load('best_weights.pt', map_location=device))
    model.eval()

    metrics = validate(model, test_loader)

    print(f"Mean Dice (no background): {metrics['mean_dice']:.4f}")
    print(f"Per-class Dice (no background): {[round(x, 4) for x in metrics['per_class_dice']]}")
    print(f"Mean HD95 (no background): {metrics['mean_hd95']:.4f}")

    print("done")
    
if __name__ == "__main__":
    main()