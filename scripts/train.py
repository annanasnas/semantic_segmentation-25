from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
from scripts.utils import fast_hist, per_class_iou, visualize_result, denormalize
from pathlib import Path
import matplotlib.pyplot as plt
from IPython.display import display


CITYSCAPES_CLASSES = [
    'Road', 'Sidewalk', 'Building', 'Wall', 'Fence', 'Pole', 'Traffic light',
    'Traffic sign', 'Vegetation', 'Terrain', 'Sky', 'Person', 'Rider', 'Car',
    'Truck', 'Bus', 'Train', 'Motorcycle', 'Bicycle'
]

CITYSCAPES_PALETTE = [
    [128, 64,128], [244, 35,232], [ 70, 70, 70], [102,102,156], [190,153,153],
    [153,153,153], [250,170, 30], [220,220,  0], [107,142, 35], [152,251,152],
    [ 70,130,180], [220, 20, 60], [255,  0,  0], [  0,  0,142], [  0,  0, 70],
    [  0, 60,100], [  0, 80,100], [  0,  0,230], [119, 11, 32]
]


def train_model(model, train_dataloader, val_dataloader,
                device, epochs, autocast, scaler,
                optimizer, criterion, scheduler, 
                ckpt, start_epoch, best_miou,
                log_csv, metrics):
    

    log_csv = Path(log_csv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    line_train, = ax1.plot([], [], label="train_loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.grid(); ax1.legend()
    line_miou,  = ax2.plot([], [], color="green", label="val_miou")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("mIoU"); ax2.grid()
    fig.suptitle("Training progress"); fig.tight_layout()
    plot_handle = display(fig, display_id="loss_miou")

    for epoch in range(start_epoch, epochs):

        model.train()
        running_loss = 0.0

        for images, masks in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
            images, masks = images.to(device), masks.to(device, dtype=torch.long)
            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                outputs, _, _ = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        train_loss = running_loss / len(train_dataloader)
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}")

        model.eval()
        hist = np.zeros((19, 19))

        with torch.no_grad():
            for images, masks in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                images, masks = images.to(device), masks.to(device, dtype=torch.long)

                with autocast(device_type='cuda'):
                    outputs = model(images)

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                targets = masks.cpu().numpy()
                targets[targets == 255] = -1
                for lt, lp in zip(targets, preds):
                    hist += fast_hist(lt.flatten(), lp.flatten(), 19)
        
        ious = per_class_iou(hist)
        miou = np.nanmean(ious)
        print("Per-class IoU")
        for idx, (cls_name, iou) in enumerate(zip(CITYSCAPES_CLASSES, ious)):
            print(f"{cls_name:16s}: {iou:.4f}")
        print(f"Mean IoU: {np.nanmean(ious):.4f}")

        snapshot = {
              "epoch":      epoch + 1,
              "model":      model.cpu().state_dict(),
              "optimizer":  optimizer.state_dict(),
              "scaler":     scaler.state_dict(),
              "scheduler":  scheduler.state_dict(),
              "best_miou":  ckpt.best,
          }
        ckpt.save_last(snapshot)
        model.to(device)

        if miou > best_miou:
            best_miou = miou
            ckpt.save_best(miou, snapshot)

        new_row = {
            "epoch":      epoch + 1,
            "train_loss": train_loss,
            "val_miou":   miou,
        }
        metrics["epoch"].append(new_row["epoch"])
        metrics["train_loss"].append(new_row["train_loss"])
        metrics["val_miou"].append(new_row["val_miou"])
        pd.DataFrame([new_row]).to_csv(
            log_csv, mode="a", index=False, header=not log_csv.exists()
        )

        line_train.set_data(metrics["epoch"], metrics["train_loss"])
        line_miou.set_data(metrics["epoch"], metrics["val_miou"])

        for ax in (ax1, ax2):
            ax.relim()
            ax.autoscale_view()

        plot_handle.update(fig)

    ckpt.save_final(snapshot, epochs)


