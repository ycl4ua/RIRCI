from __future__ import absolute_import

import torch
import torch.nn.functional as F

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor

def resize_to_match(fm,to):
    # just use interpolate
    # [1,3] = (h,w)
    return F.interpolate(fm,to.size()[-2:],mode='bilinear',align_corners=False)

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def adjust_learning_rate(datasets, model, epoch, lr,args):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in args.schedule:
        lr *= args.gamma
        optimizers = [getattr(model.model, attr) for attr in dir(model.model) if  attr.startswith("optimizer") and getattr(model.model, attr) is not None]
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    else: # access the current learning rate
        optimizers = [getattr(model.model, attr) for attr in dir(model.model) if  attr.startswith("optimizer") and getattr(model.model, attr) is not None]
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
    # decay sigma
    # for dset in datasets:
    #     if args.sigma_decay > 0:
    #         dset.dataset.sigma *=  args.sigma_decay
    #         dset.dataset.sigma *=  args.sigma_decay

    return lr




