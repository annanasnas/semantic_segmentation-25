import numpy as np
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
import time
import pandas as pd
from pathlib import Path


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=300, power=0.9):
    """Polynomial decay of learning rate
            :param init_lr is base learning rate
            :param iter is a current iteration
            :param lr_decay_iter how frequently decay occurs, default is 1
            :param max_iter is number of maximum iterations
            :param power is a polymomial power

    """

    lr = init_lr*(1 - iter/max_iter)**power
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr * 10
    return lr


def fast_hist(a, b, n):
    '''
    a and b are label and prediction respectively
    n is the number of classes
    '''
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iou(hist):
    epsilon = 1e-5
    return (np.diag(hist)) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def latency_FPS(model, device, h, w):
    model.eval().to(device) 
    image = torch.randn(1, 3, h, w).to(device)
    iterations = 1000
    latency = []
    FPS = []

    for i in range(iterations):
        start = time.time()
        output = model(image)
        end = time.time()

        latencyi = end - start

        if latencyi != 0.0:
            latency.append(latencyi)
            FPSi = 1/latencyi
            FPS.append(FPSi)

    mean_latency = np.mean(latency)
    std_latency = np.std(latency)
    mean_FPS = np.mean(FPS)
    std_FPS = np.std(FPS)

    return mean_latency, std_latency, mean_FPS, std_FPS


def create_final_table(model, model_name, device, in_res, epochs):
    h, w = in_res
    model = model.to(device)

    latency_mean, latency_std, fps_mean, fps_std = latency_FPS(model, device, h, w)
    flops = FlopCountAnalysis(model, torch.zeros(1,3,h,w,device=device)).total()/1e9
    params_m = count_params(model)/1e6

    df = pd.DataFrame(
        [[f"{model_name} - {epochs} epochs",
          f"Mean latency: {latency_mean:.2f} +/- {latency_std:.2f}, Mean FPS: {fps_mean:.2f} +/- {fps_std:.2f} frames per second",
          f"{flops:.1f} G",
          f"{params_m:.1f} M"]],
        columns=["Model", "Latency", "FLOPs", "Params"]
    )
    return df


def decode_segmap(label, CITYSCAPES_PALETTE):
    rgb = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for cls_id in range(19):
        rgb[label == cls_id] = CITYSCAPES_PALETTE[cls_id]
    return rgb


def denormalize(image):
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    return image * std + mean


def plot_log(csv_path):
    df = pd.read_csv(csv_path).sort_values("epoch")

    fig, (ax_loss, ax_miou) = plt.subplots(
        nrows=2, ncols=1, figsize=(8, 8), sharex=True
    )

    ax_loss.plot(df["epoch"], df["train_loss"], color="tab:blue", label="train loss")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training Loss")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(frameon=False)

    ax_miou.plot(df["epoch"], df["val_miou"], color="tab:green", label="val mIoU")
    ax_miou.set_xlabel("Epoch")
    ax_miou.set_ylabel("mIoU")
    ax_miou.set_title("Validation mIoU")
    ax_miou.grid(alpha=0.3)
    ax_miou.legend(frameon=False)

    fig.tight_layout()
    plt.show()

def visualize_sample(val_dataloader, model):

    CITYSCAPES_PALETTE = [
        [128, 64,128], [244, 35,232], [ 70, 70, 70], [102,102,156], [190,153,153],
        [153,153,153], [250,170, 30], [220,220,  0], [107,142, 35], [152,251,152],
        [ 70,130,180], [220, 20, 60], [255,  0,  0], [  0,  0,142], [  0,  0, 70],
        [  0, 60,100], [  0, 80,100], [  0,  0,230], [119, 11, 32]
    ]
    image, label = next(iter(val_dataloader))
    image = image[0]
    label = label[0].cpu().numpy()

    model.eval()
    with torch.no_grad():
        logits = model(image.unsqueeze(0))
        pred   = logits.argmax(1).squeeze().cpu().numpy()

    image_vis = TF.to_pil_image(denormalize(image.cpu()))

    gt_rgb   = decode_segmap(label, CITYSCAPES_PALETTE)
    pred_rgb = decode_segmap(pred,  CITYSCAPES_PALETTE)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(image_vis); axs[0].set_title("Input image")
    axs[1].imshow(gt_rgb);    axs[1].set_title("Ground truth")
    axs[2].imshow(pred_rgb);  axs[2].set_title("Prediction")

    for ax in axs: ax.axis("off")
    plt.tight_layout(); plt.show()

