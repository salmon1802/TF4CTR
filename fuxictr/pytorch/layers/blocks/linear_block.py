# -*- coding: utf-8 -*-
# @Time    : 2023/4/17 12:19
# @Author  : salmon1802
# @Software: PyCharm

import torch
from torch import nn
from fuxictr.pytorch.torch_utils import get_activation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Linear_Block(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 output_activation="leaky_relu",
                 use_bias=False):
        super(Linear_Block, self).__init__()
        output_activation = get_activation(output_activation)
        self.Linear = nn.Sequential(nn.Linear(input_dim, output_dim, bias=use_bias),
                                    output_activation)
        self.to(device)

    def forward(self, inputs):
        return self.Linear(inputs)
