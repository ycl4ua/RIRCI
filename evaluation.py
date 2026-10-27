import numpy as np
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_RMSE(pred, gt, mask, is_w=False):
    if is_w:
        mse = torch.mean((pred * mask - gt * mask) ** 2, dim=[1, 2, 3])
        rmse = mse * np.prod(mask.shape[1:]) / (torch.sum(mask, dim=[1, 2, 3]) + 1e-6)
        rmse = torch.sqrt(rmse).mean().item()
    else:
        mse = torch.mean((pred - gt) ** 2, dim=[1, 2, 3])
        rmse = torch.sqrt(mse).mean().item()

    return rmse * 256


def compute_IoU(pred, gt, threshold=0.5, eps=1e-5):
    pred = torch.where(pred > threshold, torch.ones_like(pred), torch.zeros_like(pred)).to(pred.device)
    intersection = (pred * gt).sum(dim=[1, 2, 3])
    union = pred.sum(dim=[1, 2, 3]) + gt.sum(dim=[1, 2, 3]) - intersection
    return (intersection / (union + eps)).mean().item()


def FScore(pred, gt, beta2=1.0, threshold=0.5, eps=1e-6, reduce_dims=[1, 2, 3]):
    if isinstance(pred, torch.Tensor):
        if threshold == -1:
            threshold = pred.mean().item() * 2
        ones = torch.ones_like(pred).to(pred.device)
        zeros = torch.zeros_like(pred).to(pred.device)
        pred_ = torch.where(pred > threshold, ones, zeros)
        gt = torch.where(gt > threshold, ones, zeros)

        TP = (pred_ * gt).sum(dim=reduce_dims)
        NumPrecision = pred_.sum(dim=reduce_dims)
        NumRecall = gt.sum(dim=reduce_dims)

        precision = TP / (NumPrecision + eps)
        recall = TP / (NumRecall + eps)
        F_beta = (1 + beta2) * (precision * recall) / (beta2 * precision + recall + eps)
        F_beta = F_beta.mean()

    elif isinstance(pred, np.ndarray):
        if threshold == -1:
            threshold = pred.mean() * 2
        pred_ = np.where(pred > threshold, 1.0, 0.0)
        gt = np.where(gt > threshold, 1.0, 0.0)

        TP = (pred_ * gt).sum()
        NumPrecision = pred_.sum()
        NumRecall = gt.sum()

        precision = TP / (NumPrecision + eps)
        recall = TP / (NumRecall + eps)
        F_beta = (1 + beta2) * (precision * recall) / (beta2 * precision + recall + eps)

    return F_beta
