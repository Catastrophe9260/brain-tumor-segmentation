import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from monai.losses import DiceLoss # 1 - (2 * |P ∩ T|) / (|P| + |T|), checks overlap and isn't biased by background
import mlflow
import mlflow.pytorch
from mlflow import log_param, log_artifacts, log_metric

from model import ResAtt3DUNet
from utils import npy_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():

    print("starting train loop...")

    with mlflow.start_run(run_name="brainseg_1"):

        # hyperparameters
        use_data_aug = True
        in_ch = 3
        out_ch = 4
        num_filters = 32
        num_heads = 2
        use_dropout = True
        dropout_prob = 0.2
        batch_size = 2
        lr = 1e-4
        sched_factor = 0.5
        sched_patience = 20
        num_epochs = 250
        log_every = 25

        # log hyperparameters and model information
        log_param("dataset", "BraTS2020")
        log_param("data_augmentation", use_data_aug)
        log_param("model", "ResAtt3DUNet")
        log_param("in_channels", in_ch)
        log_param("out_channels", out_ch)
        log_param("num_filters", num_filters)
        log_param("num_heads", num_heads)
        log_param("dropout", use_dropout)
        log_param("dropout_probability", dropout_prob)
        log_param("batch_size", batch_size)
        log_param("optimizer", "Adam")
        log_param("learning_rate", lr)
        log_param("scheduling_factor", sched_factor)
        log_param("scheduling_patience", sched_patience)
        log_param("loss_function", "DiceLoss + CrossEntropyLoss")
        log_param("num_epochs", num_epochs)

        # data setup
        data_dir = '/home/omkos333/projects/brain_tumor_seg/data/processed'

        train_ds = npy_dataset(
            os.path.join(data_dir, 'train', 'images'),
            os.path.join(data_dir, 'train', 'masks'),
            data_augmentation=use_data_aug
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        # model setup
        model = ResAtt3DUNet(
            in_channels=in_ch, out_channels=out_ch, num_filters=num_filters, num_heads=num_heads,
            dropout=use_dropout, dropout_probability=dropout_prob
        )
        
        model = model.to(device)
        
        # optimizer with a learning rate scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=sched_factor, patience=sched_patience
        )

        # loss function
        class_weights = torch.Tensor([0.1, 1.0, 1.0, 1.0]).to(device) # reduce importance of background

        loss_fn = lambda logits, targets: (
            DiceLoss()(logits, targets) + nn.CrossEntropyLoss(weight=class_weights)(logits, torch.argmax(targets, dim=1))
        )

        # training loop
        for epoch in range(1, num_epochs + 1):
            model.train()

            epoch_loss = 0
            num_batches = 0

            for img, true_mask in train_loader:
                img, true_mask = img.to(device), true_mask.to(device)

                logits = model(img)
                loss = loss_fn(logits, true_mask)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches

            # log per-epoch metrics
            log_metric("train_loss", avg_loss, step=epoch)
            log_metric("scheduled_learning_rate", optimizer.param_groups[0]['lr'], step=epoch)

            print(f"epoch {epoch}, loss: {avg_loss:.4f}, lr: {optimizer.param_groups[0]['lr']:.6f}")

            scheduler.step(avg_loss)

            # log model periodically
            if epoch % log_every == 0:
                mlflow.pytorch.log_model(model, name=f"epoch_{epoch}_model")
                log_artifacts(f'epoch_{epoch}_weights.pt', artifact_path="model_weights")

        # save final model weights
        torch.save(model.state_dict(), 'final_weights.pt')
        
        # log final model
        mlflow.pytorch.log_model(model, name="final_model")
        log_artifacts('final_weights.pt', artifact_path="model_weights")

        print("done")

if __name__ == "__main__":
    main()