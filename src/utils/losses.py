"""Loss components used by RIRCI (Eq. 2-10 of the paper)."""

import torch
import torch.nn as nn
from torchvision import models


def l1_relative(reconstructed, real, mask):
    """Masked L1 loss normalised by the mask area (Eq. 3)."""
    batch = real.size(0)
    area = torch.sum(mask.view(batch, -1), dim=1)
    reconstructed = reconstructed * mask
    real = real * mask

    loss_l1 = torch.abs(reconstructed - real).view(batch, -1)
    loss_l1 = torch.sum(loss_l1, dim=1) / (area + 1e-6)
    loss_l1 = torch.sum(loss_l1) / batch
    return loss_l1


def is_dic(x):
    return type(x) == type([])


class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        """norm (bool): normalize/denormalize the stats"""
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


class VGGLossX(nn.Module):
    """Perceptual loss on three VGG16 feature levels (Eq. 4)."""

    def __init__(self, normalize=True):
        super(VGGLossX, self).__init__()
        self.vgg = VGG16FeatureExtractor().cuda()
        self.criterion = nn.L1Loss().cuda()
        if normalize:
            self.normalize = MeanShift([0.485, 0.456, 0.406],
                                       [0.229, 0.224, 0.225], norm=True).cuda()
        else:
            self.normalize = None

    def forward(self, x, y):
        if self.normalize is not None:
            x = self.normalize(x)
            y = self.normalize(y)

        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        loss = 0
        for i in range(3):
            loss += self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


def VGGLoss():
    return VGGLossX()
