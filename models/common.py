import torch
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

def reparameterize_gaussian(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn(std.size()).to(mean)
    return mean + std * eps


def gaussian_entropy(logvar):
    const = 0.5 * float(logvar.size(1)) * (1. + np.log(np.pi * 2))
    ent = 0.5 * logvar.sum(dim=1, keepdim=False) + const
    return ent


def standard_normal_logprob(z):
    dim = z.size(-1)
    log_z = -0.5 * dim * np.log(2 * np.pi)
    return log_z - z.pow(2) / 2


def truncated_normal_(tensor, mean=0, std=1, trunc_std=2):
    """
    Taken from https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < trunc_std) & (tmp > -trunc_std)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def get_linear_scheduler(optimizer, start_epoch, end_epoch, start_lr, end_lr):
    def lr_func(epoch):
        if epoch <= start_epoch:
            return 1.0
        elif epoch <= end_epoch:
            total = end_epoch - start_epoch
            delta = epoch - start_epoch
            frac = delta / total
            return (1-frac) * 1.0 + frac * (end_lr / start_lr)
        else:
            return end_lr / start_lr
    return LambdaLR(optimizer, lr_lambda=lr_func)

def get_cosine_scheduler(optimizer, start_epoch, end_epoch, start_lr, end_lr):
    """
    Not used in this code - makes the function between the start 
    and end epoch of of the learning rate like a cos function instead of 
    a linear one like used in the original work. As I did not look at LR 
    adjustments, it went unused.
    """
    def lr_func(epoch):
        if epoch < start_epoch:
            return 1.0
        
        elif epoch <= end_epoch:
            amplitude = (1 - end_lr / start_lr)/2
            frequency = 1/((end_epoch - start_epoch)*2)
            t = epoch-start_epoch
            offset = (1 - end_lr / start_lr)/2 + end_lr / start_lr
            return offset + amplitude * np.cos(frequency*2*np.pi*t)
            
        else:
            return end_lr / start_lr

    return LambdaLR(optimizer, lr_lambda=lr_func)

