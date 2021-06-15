import torch
import torch.nn as nn


class Sinkhorn(nn.Module):
    """
    BiStochastic Layer turns the input matrix into a bi-stochastic matrix.
    Parameter: maximum iterations max_iter
               a small number for numerical stability epsilon
    Input: input matrix cost
    Output: bi-stochastic matrix cost
    """
    def __init__(self, max_iter=10, epsilon=1e-4):
        super(Sinkhorn, self).__init__()
        self.max_iter = max_iter
        self.epsilon = epsilon

    def forward(self, costs, exp=False, exp_alpha=20, dummy_row=False, dtype=torch.float32):
        ns_src = costs.shape[0]
        ns_tgt = costs.shape[1]

        row_norm_ones = torch.zeros(ns_src, ns_src, device=costs.device)  # size: row x row
        col_norm_ones = torch.zeros(ns_tgt, ns_tgt, device=costs.device)  # size: col x col
        row_slice = slice(0, ns_tgt)
        col_slice = slice(0, ns_src)
        row_norm_ones[row_slice, row_slice] = 1
        col_norm_ones[col_slice, col_slice] = 1

        costs += self.epsilon

        for i in range(self.max_iter):
            if exp:
                costs = torch.exp(exp_alpha * costs)
            if i % 2 == 1:
                # column norm
                sum = torch.sum(torch.mul(costs, col_norm_ones), dim=1)
            else:
                # row norm
                sum = torch.sum(torch.mul(row_norm_ones, costs), dim=1)

            tmp = torch.zeros_like(costs)
            row_slice = slice(0, ns_tgt)
            col_slice = slice(0, ns_src)
            tmp[row_slice, col_slice] = 1 / sum[row_slice, col_slice]
            costs = costs * tmp

        return costs
