import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import math


class LossBinary:
    """
    Loss defined as \alpha BCE - (1 - \alpha) SoftJaccard
    """

    def __init__(self, jaccard_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss()

    def __call__(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)

        return loss
        
