"""
Edited code to handle mish_cuda in CPU for simple inference
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def mish(input):
    return input * torch.tanh(F.softplus(input))

class MishCuda(nn.Module):
    def __init__(self):
        super().__init__
    def forward(self, input):
        return mish(input)