import numpy as np
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
import time
import pandas as pd


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
def evaluate_miou(model, dataloader, device):
    model.eval()
    hist = np.zeros((19, 19))
    for imgs, masks in dataloader:
        imgs  = imgs.to(device)
        masks = masks.to(device, dtype=torch.long)
        out   = model(imgs)[0] if isinstance(model(imgs), tuple) else model(imgs)
        preds = torch.argmax(out, 1).cpu().numpy()
        targs = masks.cpu().numpy(); targs[targs == 255] = -1
        for lt, lp in zip(targs, preds):
            hist += fast_hist(lt.flatten(), lp.flatten(), 19)
    return float(np.nanmean(per_class_iou(hist)))


@torch.no_grad()
def latency_FPS(model, device, h, w):
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


