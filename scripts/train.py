from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
from scripts.utils import fast_hist, per_class_iou, denormalize, poly_lr_scheduler
from pathlib import Path
import matplotlib.pyplot as plt
from IPython.display import display
import torch.nn.functional as F
from torch.amp import autocast


def validate(model, val_dataloader, device):

    CITYSCAPES_CLASSES = [
        'Road', 'Sidewalk', 'Building', 'Wall', 'Fence', 'Pole', 'Traffic light',
        'Traffic sign', 'Vegetation', 'Terrain', 'Sky', 'Person', 'Rider', 'Car',
        'Truck', 'Bus', 'Train', 'Motorcycle', 'Bicycle'
    ]

    model.eval()
    model.to(device)
    
    hist = np.zeros((19, 19))

    with torch.no_grad():
        for images, masks in tqdm(val_dataloader, desc=f"Validation...", leave=False):
            images, masks = images.to(device).float(), masks.to(device, dtype=torch.long)

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

    return miou


def train_deeplab(start_epoch, epochs, model, 
                  train_dataloader, val_dataloader, device,
                  optimizer, criterion, scaler, 
                  learning_rate, max_iter, iteration,
                  best_miou, metrics, ckpt, log_csv):
    
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

            poly_lr_scheduler(optimizer, learning_rate, iteration, max_iter=max_iter)
            iteration += 1

        train_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}")

        miou = validate(model, val_dataloader, device)

        #save log
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

        # visualize
        line_train.set_data(metrics["epoch"], metrics["train_loss"])
        line_miou.set_data(metrics["epoch"], metrics["val_miou"])

        for ax in (ax1, ax2):
            ax.relim()
            ax.autoscale_view()

        plot_handle.update(fig)

        #save checkpoint
        snapshot = {
              "epoch":      epoch + 1,
              "model":      model.cpu().state_dict(),
              "optimizer":  optimizer.state_dict(),
              "scaler":     scaler.state_dict(),
              "iteration":  iteration,
              "best_miou":  best_miou
          }
        ckpt.save_last(snapshot)
        model.to(device)

        if miou > best_miou:
            best_miou = miou
            ckpt.save_best(miou, snapshot)

    ckpt.save_final(snapshot, epochs)


def train_bisenet(start_epoch, epochs, model, 
                  train_dataloader, val_dataloader, device,
                  optimizer, criterion, scaler, 
                  learning_rate, max_iter, iteration,
                  best_miou, metrics, ckpt, log_csv):
    

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
                outputs, aux1, aux2 = model(images)
                loss_main = criterion(outputs, masks)
                loss_aux1 = criterion(aux1, masks)
                loss_aux2 = criterion(aux2, masks)
                loss = loss_main + 0.4 * loss_aux1 + 0.4 * loss_aux2

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            poly_lr_scheduler(optimizer, learning_rate, iteration, max_iter=max_iter)
            iteration += 1

        train_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}")

        miou = validate(model, val_dataloader, device)

        #save log
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

        # visualize
        line_train.set_data(metrics["epoch"], metrics["train_loss"])
        line_miou.set_data(metrics["epoch"], metrics["val_miou"])

        for ax in (ax1, ax2):
            ax.relim()
            ax.autoscale_view()

        plot_handle.update(fig)

        #save checkpoint
        snapshot = {
              "epoch":      epoch + 1,
              "model":      model.cpu().state_dict(),
              "optimizer":  optimizer.state_dict(),
              "scaler":     scaler.state_dict(),
              "iteration":  iteration,
              "best_miou":  best_miou
          }
        ckpt.save_last(snapshot)
        model.to(device)

        if miou > best_miou:
            best_miou = miou
            ckpt.save_best(miou, snapshot)

    ckpt.save_final(snapshot, epochs)


def train_bisenet_FDA(model, train_dataloader, val_dataloader,
                      device, epochs, autocast, scaler,
                      optimizer, criterion, learning_rate,
                      iteration, max_iter,
                      ckpt, start_epoch, best_miou,
                      log_csv, metrics,
                      charbonnier_eps, charbonnier_alpha, lambda_ent):
    

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
        total_ce = 0.0
        total_ent = 0.0

        for images, masks, target_images in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
            images, masks = images.to(device), masks.to(device, dtype=torch.long)
            target_images = target_images.to(device)
            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                outputs, aux1, aux2 = model(images)
                loss_main = criterion(outputs, masks)
                loss_aux1 = criterion(aux1, masks)
                loss_aux2 = criterion(aux2, masks)
                loss_ce = loss_main + 0.4 * loss_aux1 + 0.4 * loss_aux2

                logits_t = model(target_images)[0]
                probs_t = F.softmax(logits_t, dim=1)
                safe_probs = probs_t.clamp(min=1e-8)
                H = -torch.sum(safe_probs * torch.log(safe_probs), dim=1)
                loss_ent = torch.mean((H**2 + charbonnier_eps**2)**charbonnier_alpha)
                loss = loss_ce + lambda_ent * loss_ent

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            total_ce += loss_ce.item()
            total_ent += loss_ent.item()

            poly_lr_scheduler(optimizer, learning_rate, iteration, max_iter=max_iter)
            iteration += 1

        train_loss = running_loss / len(train_dataloader)
        avg_ce = total_ce / len(train_dataloader)
        avg_ent = total_ent / len(train_dataloader)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_dataloader):.4f}, — CE: {avg_ce:.4f}  Ent: {avg_ent:.4f}")

        miou, metrics = validate(model, autocast, val_dataloader, 
				device, epoch, epochs, 
				train_loss, metrics, log_csv)

        line_train.set_data(metrics["epoch"], metrics["train_loss"])
        line_miou.set_data(metrics["epoch"], metrics["val_miou"])

        for ax in (ax1, ax2):
            ax.relim()
            ax.autoscale_view()

        plot_handle.update(fig)

        snapshot = {
              "epoch":      epoch + 1,
              "model":      model.cpu().state_dict(),
              "optimizer":  optimizer.state_dict(),
              "scaler":     scaler.state_dict(),
              "iteration":  iteration,
              "best_miou":  best_miou
          }
        ckpt.save_last(snapshot)
        model.to(device)

        if miou > best_miou:
            best_miou = miou
            ckpt.save_best(miou, snapshot)

    ckpt.save_final(snapshot, epochs)

