import os
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from monai.metrics import DiceMetric, HausdorffDistanceMetric
import mlflow
import mlflow.pytorch
from mlflow import log_param, log_metric

from model import ResAtt3DUNet
from hyperparameters import BATCH_SIZE, IN_CH, OUT_CH, NUM_FILTERS, NUM_HEADS
from utils import npy_dataset, one_hot_from_logits

device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)

def main():

    print("starting test loop...")

    with mlflow.start_run(run_name="brainseg_test"):

        # hyperparameters
        batch_size = BATCH_SIZE
        in_ch = IN_CH
        out_ch = OUT_CH
        num_filters = NUM_FILTERS
        num_heads = NUM_HEADS

        # log data/model information and hyperparameters
        log_param("dataset", "BraTS2020")
        log_param("batch_size", batch_size)
        log_param("model", "ResAtt3DUNet")
        log_param("in_channels", in_ch)
        log_param("out_channels", out_ch)
        log_param("num_filters", num_filters)
        log_param("num_heads", num_heads)

        # data setup
        data_dir = '/home/omkos333/projects/brainseg/data/processed'

        test_ds = npy_dataset(
            os.path.join(data_dir, 'test', 'images'),
            os.path.join(data_dir, 'test', 'masks'),
            data_augmentation=False
        )

        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # model setup
        model = ResAtt3DUNet(
            in_channels=in_ch, out_channels=out_ch, num_filters=num_filters, num_heads=num_heads, dropout=False
        )

        model.load_state_dict(torch.load('final_weights.pt', map_location=device))
        model = model.to(device)
        
        mlflow.pytorch.log_model(model, "test_model")

        # define metrics
        mean_dice_metric = DiceMetric(include_background=False, reduction="mean")
        mean_hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")

        with torch.no_grad():
            model.eval()

            for image, true_mask in test_loader:
                image, true_mask = image.to(device), true_mask.to(device)

                with autocast(device_str):
                    logits = model(image)
                    pred_mask = one_hot_from_logits(logits, out_ch)

                mean_dice_metric(pred_mask, true_mask)
                mean_hd95_metric(pred_mask, true_mask)
        
        # calculate metrics
        mean_dice_score = mean_dice_metric.aggregate().item()
        mean_hd95 = mean_hd95_metric.aggregate().item()

        # reset metrics
        mean_dice_metric.reset()
        mean_hd95_metric.reset()

        # log metrics
        log_metric("mean_dice_score", mean_dice_score)
        log_metric("mean_hd95", mean_hd95)

    print("done")
    
if __name__ == "__main__":
    main()