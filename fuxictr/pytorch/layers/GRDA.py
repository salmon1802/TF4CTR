# -*- coding: utf-8 -*-
# @Time    : 2023/3/9 13:50
# @Author  : salmon1802
# @Software: PyCharm


import torch
from torch.optim.optimizer import Optimizer


class GRDA(Optimizer):
    def __init__(self,
                 params,
                 lr=0.01,
                 c=0.005,
                 mu=0.7,
                 name="GRDA"):
        """Construct a new GRDA optimizer.
            Args:
                learning_rate: A Tensor or a floating point value. The learning rate.
                c: A float value or a constant float tensor. Turn on/off the l1 penalty and initial penalty.
                mu: A float value or a constant float tensor. Time expansion of l1 penalty.
                name: Optional name for the operations created when applying gradients.
                Defaults to "GRDA".

            There are three hyperparameters: Learning rate (lr), sparsity control mu (mu), and initial sparse control constant (c) in gRDA optimizer.

            lr: as a rule of thumb, use the learning rate that works for the SGD without momentum. Scale the learning rate with the batch size.
            mu: 0.5 < mu < 1. Greater mu will make the parameters more sparse. Selecting it in the set {0.501,0.51,0.55} is generally recommended.
            c: a small number, e.g. 0 < c < 0.005. Greater c causes the src to be more sparse, especially at the early stage of training.
            c usually has small effect on the late stage of training.
            We recommend to first fix mu, then search for the largest c that preserves the testing accuracy with 1-5 epochs.
        """
        defaults = dict(lr=lr, c=c, mu=mu, name=name)
        super(GRDA, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(GRDA, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            c = group['c']
            mu = group['mu']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                param_state = self.state[p]

                if 'iter_num' not in param_state:
                    iter_num = param_state['iter_num'] = torch.zeros(1)
                    accumulator = param_state['accumulator'] = torch.FloatTensor(p.shape).to(p.device)
                    l1_accumulation = param_state['l1_accumulation'] = torch.zeros(1)
                    accumulator.data = p.clone()

                else:
                    iter_num = param_state['iter_num']
                    accumulator = param_state['accumulator']
                    l1_accumulation = param_state['l1_accumulation']
                iter_num.add_(1)
                #accumulator.data.add_(-lr, d_p)
                accumulator.data.add_(d_p, alpha=-lr)

                # l1 = c * torch.pow(torch.tensor(lr), 0.5 + mu) * torch.pow(iter_num, mu)
                l1_diff = c * torch.pow(torch.tensor(lr), mu + 0.5) * torch.pow(iter_num, mu) - c * torch.pow(
                    torch.tensor(lr), mu + 0.5) * torch.pow(iter_num - 1, mu)
                l1_accumulation += l1_diff

                new_a_l1 = torch.abs(accumulator.data) - l1_accumulation.to(p.device)
                p.data = torch.sign(accumulator.data) * new_a_l1.clamp(min=0)

        return loss