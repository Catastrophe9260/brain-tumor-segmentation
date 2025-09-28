import os
import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from monai.losses import DiceFocalLoss
import mlflow
import mlflow.pytorch
from mlflow import log_param, log_artifacts, log_metric

from model import ResAtt3DUNet
from hyperparameters import (
    USE_DATA_AUG, BATCH_SIZE, SHUFFLE,
    IN_CH, OUT_CH, NUM_FILTERS, NUM_HEADS, USE_DROPOUT, DROPOUT_PROB,
    LR, BETAS, WEIGHT_DECAY, 
    SCHED_MIN_LR,
    USE_BACKGROUND, GAMMA, L_DICE, L_FOCAL,
    NUM_EPOCHS, LOG_EVERY
)
from utils import npy_dataset, compute_class_weights

device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)

def main():

    print("starting train loop...")

    with mlflow.start_run(run_name="brainseg_train"):

        # hyperparameters
        use_data_aug = USE_DATA_AUG
        batch_size = BATCH_SIZE
        shuffle = SHUFFLE
        in_ch = IN_CH
        out_ch = OUT_CH
        num_filters = NUM_FILTERS
        num_heads = NUM_HEADS
        use_dropout = USE_DROPOUT
        dropout_prob = DROPOUT_PROB
        lr = LR
        betas = BETAS
        weight_decay = WEIGHT_DECAY
        sched_min_lr = SCHED_MIN_LR
        use_background = USE_BACKGROUND
        gamma = GAMMA
        l_dice = L_DICE
        l_focal = L_FOCAL
        num_epochs = NUM_EPOCHS
        log_every = LOG_EVERY

        # log data/model information and hyperparameters
        log_param("dataset", "BraTS2020")
        log_param("data_augmentation", use_data_aug)
        log_param("batch_size", batch_size)
        log_param("shuffle", shuffle)
        log_param("model", "ResAtt3DUNet")
        log_param("in_channels", in_ch)
        log_param("out_channels", out_ch)
        log_param("num_filters", num_filters)
        log_param("num_heads", num_heads)
        log_param("dropout", use_dropout)
        log_param("dropout_probability", dropout_prob)
        log_param("optimizer", "AdamW")
        log_param("learning_rate", lr)
        log_param("betas", betas)
        log_param("weight_decay", weight_decay)
        log_param("scheduler_min_learning_rate", sched_min_lr)  
        log_param("loss_function", "DiceFocalLoss")
        log_param("include_background", use_background)
        log_param("gamma", gamma)
        log_param("lambda_dice", l_dice)
        log_param("lambda_focal", l_focal)
        log_param("num_epochs", num_epochs)

        # data setup
        data_dir = '/home/omkos333/projects/brainseg/data/processed'

        train_ds = npy_dataset(
            os.path.join(data_dir, 'train', 'images'),
            os.path.join(data_dir, 'train', 'masks'),
            data_augmentation=use_data_aug
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)

        # model setup
        model = ResAtt3DUNet(
            in_channels=in_ch, out_channels=out_ch, num_filters=num_filters, num_heads=num_heads,
            dropout=use_dropout, dropout_probability=dropout_prob
        )
        
        model = model.to(device)

        scaler = GradScaler(device_str) # for mixed precision training
        
        # optimizer with a learning rate scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=sched_min_lr)

        # loss function
        class_weights = compute_class_weights(train_loader, out_ch, use_background, device)

        loss_fn = DiceFocalLoss(
            include_background=use_background, to_onehot_y=False, softmax=True, # our masks are already one-hot
            gamma=gamma, weight=class_weights, lambda_dice=l_dice, lambda_focal=l_focal
        )

        # training loop
        for epoch in range(1, num_epochs + 1):
            model.train()

            epoch_loss = 0
            num_batches = 0

            for image, true_mask in train_loader:
                image, true_mask = image.to(device), true_mask.to(device)

                with autocast(device_str):
                    logits = model(image)
                    loss = loss_fn(logits, true_mask)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches

            # log per-epoch metrics
            log_metric("train_loss", avg_loss, step=epoch)
            log_metric("scheduled_learning_rate", optimizer.param_groups[0]['lr'], step=epoch)

            print(f"epoch {epoch}, loss: {avg_loss:.4f}, lr: {optimizer.param_groups[0]['lr']:.6f}")

            scheduler.step()

            # log model periodically
            if epoch % log_every == 0:
                mlflow.pytorch.log_model(model, name=f"epoch_{epoch}_model")
                log_artifacts(f'epoch_{epoch}_weights.pt', artifact_path="model_weights")

        # save and log final model weights
        torch.save(model.state_dict(), 'final_weights.pt')
        log_artifacts('final_weights.pt', artifact_path="model_weights")

        # log final model
        mlflow.pytorch.log_model(model, name="final_model")

    print("done")

if __name__ == "__main__":
    main()