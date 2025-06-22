import numpy as np
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt


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


def decode_segmap(label, CITYSCAPES_PALETTE):
    rgb = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for cls_id in range(19):
        rgb[label == cls_id] = CITYSCAPES_PALETTE[cls_id]
    return rgb


def denormalize(image):
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    return image * std + mean


def visualize_result(sample_image, sample_mask, sample_pred, CITYSCAPES_PALETTE):
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
