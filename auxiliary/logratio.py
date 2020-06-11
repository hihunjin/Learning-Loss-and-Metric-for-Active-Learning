from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd import Function
import torch.backends.cudnn as cudnn
from auxiliary.utils import L2dist
# from utils import L2dist

class LossToDist(nn.Module):
    def __init__(self):
        super(LossToDist, self).__init__()
        self.norm = 2
    
    def forward(self, losses):
        losses_a = losses[0]
        losses_p = losses[1:]
        eps = 1e-4
        diff = torch.abs(losses_a-losses_p)
        out = torch.pow(diff, self.norm)
        gt_dist = torch.pow(out + eps, 1. / self.norm)

        return gt_dist



class LogRatioLoss(nn.Module):
    """Log ratio loss function. """
    def __init__(self):
        super(LogRatioLoss, self).__init__()
        self.pdist = L2dist(2)  # norm 2

    def forward(self, input, gt_dist):
        m = input.size()[0]-1   # #paired
        a = input[0]            # anchor
        p = input[1:]           # paired
        
        # auxiliary variables
        idxs = torch.arange(1, m+1).cuda()
        indc = idxs.repeat(m,1).t() < idxs.repeat(m,1)

        epsilon = 1e-6

        dist = self.pdist.forward(a,p)

        log_dist = torch.log(dist + epsilon)
        log_gt_dist = torch.log(gt_dist + epsilon)
        diff_log_dist = log_dist.repeat(m,1).t()-log_dist.repeat(m, 1)
        diff_log_gt_dist = log_gt_dist.repeat(m,1).t()-log_gt_dist.repeat(m, 1)

        # uniform weight coefficients 
        wgt = indc.clone().float()
        wgt = wgt.div(wgt.sum())

        log_ratio_loss = (diff_log_dist-diff_log_gt_dist).pow(2)

        loss = log_ratio_loss
        loss = loss.mul(wgt).sum()

        return loss

def main():
    data_size = 100
    input_dim = 3
    output_dim = 512

    x = Variable(torch.rand(data_size, input_dim), requires_grad=False)
    w = Variable(torch.rand(input_dim, output_dim), requires_grad=True)
    embeddings = x.mm(w)

    losses = Variable(torch.rand(data_size), requires_grad=False)
    gt_dist = LossToDist()(losses)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    print(LogRatioLoss()(embeddings.cuda(), gt_dist.cuda()))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')

