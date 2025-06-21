

def train_model(scaler, start_epoch, epochs, train_dataloader, device):

    max_iter = len(train_dataloader) * epochs
    log_df = pd.DataFrame(columns=['epoch', 'train_loss', 'val_miou'])
    train_losses = []
    val_mious = []
    best_miou = 0.0

    for epoch in range(start_epoch, epochs):

        model.train()
        running_loss = 0.0

        train_loader_tqdm = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for images, masks in train_loader_tqdm:
            images = images.to(device)
            masks = masks.to(device, dtype=torch.long)

            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                outputs, _, _ = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            train_loader_tqdm.set_postfix(loss=loss.item())

        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_dataloader):.4f}")

        model.eval()
        hist = np.zeros((19, 19))

        val_loader_tqdm = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
        with torch.no_grad():
            for images, masks in val_loader_tqdm:
                images = images.to(device)
                masks = masks.to(device, dtype=torch.long)

                with autocast(device_type='cuda'):
                    outputs, _, _ = model(images)

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                targets = masks.cpu().numpy()

                targets[targets == 255] = -1 # delete void

                for lt, lp in zip(targets, preds):
                    hist += fast_hist(lt.flatten(), lp.flatten(), 19)

        ious = per_class_iou(hist)
        print("Per-class IoU")
        for idx, (cls_name, iou) in enumerate(zip(CITYSCAPES_CLASSES, ious)):
            print(f"{cls_name:16s}: {iou:.4f}")
        print(f"Mean IoU: {np.nanmean(ious):.4f}")

        sample_image = images[0].cpu()

        sample_image = denormalize(sample_image.clone())
        sample_mask = masks[0].cpu().numpy()
        sample_pred = preds[0]
        gt_rgb = decode_segmap(sample_mask, CITYSCAPES_PALETTE)
        pred_rgb = decode_segmap(sample_pred, CITYSCAPES_PALETTE)
        sample_image_pil = TF.to_pil_image(sample_image)

        gt_rgb = decode_segmap(sample_mask, CITYSCAPES_PALETTE)
        pred_rgb = decode_segmap(sample_pred, CITYSCAPES_PALETTE)
        sample_image_pil = TF.to_pil_image(sample_image)

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(sample_image_pil)
        axs[0].set_title("Input image")
        axs[1].imshow(gt_rgb)
        axs[1].set_title("Ground truth")
        axs[2].imshow(pred_rgb)
        axs[2].set_title("Prediction")
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

        # сохраняем результаты каждую 5 эпоху
        checkpoint_path = f"/content/drive/MyDrive/semantic segmentation/Other/checkpoint_v2_{epoch+1}.pth"
        if (epoch + 1) % 5 == 0:
          torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              #'iteration': iteration,
              'scaler_state_dict': scaler.state_dict(),
              'scheduler': scheduler.state_dict()
          }, checkpoint_path)
          print(f"Checkpoint saved at {checkpoint_path}")

        # сохранение loss и miou
        epoch_loss = running_loss / len(train_dataloader)
        epoch_miou = np.nanmean(ious)

        if epoch_miou > best_miou:
          best_miou = epoch_miou
          best_ckpt_path = "/content/drive/MyDrive/semantic segmentation/Other/deeplab_best_model.pth"
          torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              #'iteration': iteration,
              'scaler_state_dict': scaler.state_dict(),
              'scheduler': scheduler.state_dict()
          }, best_ckpt_path)
          print(f"Best model updated at epoch {epoch+1}, mIoU = {best_miou:.4f}")


        train_losses.append(epoch_loss)
        val_mious.append(epoch_miou)

        plt.figure(figsize=(10, 4))

        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)
        plt.legend()

        # mIoU
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(val_mious) + 1), val_mious, marker='s', color='green', label='Val mIoU')
        plt.xlabel('Epoch')
        plt.ylabel('mIoU')
        plt.title('Validation mIoU')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        if (epoch + 1) == 50:
          plt.savefig('/content/drive/MyDrive/semantic segmentation/Other/loss_miou_plot.png')
        plt.show()
        plt.close()

        log_df.loc[len(log_df)] = {
            'epoch': epoch + 1,
            'train_loss': epoch_loss,
            'val_miou': epoch_miou
        }

        log_df.to_csv(log_csv_path, index=False)