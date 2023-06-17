'''
Date: 2021-01-13 16:32:55
LastEditTime: 2021-01-13 20:34:50
Description: loss function for training phase
FilePath: /DMGAN/lib/loss.py
'''
import torch

def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()
