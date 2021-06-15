import torch.nn  as nn
from PCA.sinkhorn import Sinkhorn

class GraphMatchingSolver(nn.Module):
    def __init__(self, alpha=200, max_iter=10, epsilon=1e-4):
        super().__init__()
        self.bi_stochastic = Sinkhorn(max_iter=max_iter, epsilon=epsilon)
        self.alpha = alpha
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, unary_costs_list):
        res = []
        for unary_cost in unary_costs_list:
            s = self.softmax(self.alpha * unary_cost)
            s = self.bi_stochastic(s)
            res.append(s)
        return res
