import random

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from data.sampler import SubsetSequentialSampler

from strategy.entropysampling import EntropySampling
from strategy.marginsampling import MarginSampling
from strategy.least_confidence import LeastConfidence
from strategy.bayesian_active_learning_disagreement_dropout import BALDDropout

from debug_config import *

criterion      = nn.CrossEntropyLoss(reduction='none')

def get_uncertainty(models, unlabeled_loader):
    models['backbone'].eval()
    models['module'].eval()
    uncertainty = torch.tensor([]).cuda()
    real_loss = torch.tensor([]).cuda()

    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()

            scores, features = models['backbone'](inputs)
            pred_loss = models['module'](features)[0]
            real_losst = criterion(scores, labels) # ground truth loss
            pred_loss = pred_loss.view(pred_loss.size(0))
            
            real_loss = torch.cat((real_loss,real_losst),0)
            uncertainty = torch.cat((uncertainty, pred_loss), 0)
    
    return uncertainty.cpu(), real_loss.cpu()


def Sampler(rule, models, cifar10_unlabeled, unlabeled_set):

    # Randomly sample 10000 unlabeled data points
    random.shuffle(unlabeled_set)
    if rule in ['PredictedLoss', 'BALDDropout']:
        subset = unlabeled_set[:SUBSET]
    else:
        subset = unlabeled_set

    # Create unlabeled dataloader for the unlabeled subset
    unlabeled_loader = DataLoader(cifar10_unlabeled, batch_size=BATCH, 
                                    sampler=SubsetSequentialSampler(subset), # more convenient if we maintain the order of subset
                                    pin_memory=True)



    # Measure uncertainty of each data points in the subset
    if rule == 'PredictedLoss':
        uncertainty, real_loss = get_uncertainty(models, unlabeled_loader)
        # uncertainty = get_uncertainty(models, unlabeled_loader)
        return (uncertainty, real_loss, subset)
    elif rule == 'Entropy':
        uncertainty = EntropySampling(models, unlabeled_loader)
    elif rule == 'Random':
        uncertainty = torch.rand(len(subset))
    elif rule == 'Margin':
        uncertainty = MarginSampling(models, unlabeled_loader)
    elif rule == 'LeastConfidence':
        uncertainty = LeastConfidence(models, unlabeled_loader)
    elif rule == 'BALDDropout':
        uncertainty = BALDDropout(models, unlabeled_loader)
    
    return (uncertainty, None, subset)