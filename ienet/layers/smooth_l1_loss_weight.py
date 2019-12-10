# -*- coding: utf-8 -*-


import torch
from torch import nn


class smooth_l1_loss_weight(nn.Module):

    def __init__(self, beta=1. / 9):
        super(smooth_l1_loss_weight, self).__init__()
        self.beta = beta

    def forward(self, input, target, weight=None):
        """
        very similar to the smooth_l1_loss from pytorch, but with
        the extra beta parameter
        """
        n = torch.abs(input - target)
        cond = n < self.beta
        loss = torch.where(cond, 0.5 * n ** 2 / self.beta, n - 0.5 * self.beta)
#        loss = torch.log(loss)
        if weight is not None and weight.sum() > 0:
                return (loss * weight[:, None]).sum()
        else:
            assert loss.numel() != 0
            return loss.sum()
        
#    return loss.sum()
