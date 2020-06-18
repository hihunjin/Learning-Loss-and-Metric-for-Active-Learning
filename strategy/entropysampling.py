import torch
import sys
sys.path.append('..')
from debug_config import *
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from data.sampler import SubsetSequentialSampler
import torch.nn.functional as F

def predict_prob(models, unlabeled_loader):
    models['backbone'].eval()
    models['module'].eval()
    probs = torch.tensor([]).cuda()

    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()
            out = models['backbone'](inputs)[0]
            prob = F.softmax(out, dim=1)
            probs = torch.cat((probs,prob),0)

    return probs.cpu()

def EntropySampling(model, loader):
    probs = predict_prob(model, loader)
    log_probs = torch.log(probs)
    U = -(probs*log_probs).sum(1)

    return U



if __name__== '__main__' :
    from torchvision.datasets import CIFAR10
    import models.resnet as resnet
    import torchvision.transforms as T

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]) # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
    ])

    cifar10_unlabeled   = CIFAR10('../cifar10', train=True, download=True, transform = test_transform)


    models={}
    models['backbone'] = resnet.ResNet18(num_classes=10).cuda()
    models['module'] = resnet.ResNet18(num_classes=10).cuda()
    labeled_set = [i for i in range(4000)]
    unlabeled_set = [i for i in range(4000,40000)]
    subset = unlabeled_set[:SUBSET]
    loader = DataLoader(cifar10_unlabeled, batch_size=BATCH,
        sampler=SubsetSequentialSampler(subset), # more convenient if we maintain the order of subset
        pin_memory=True)

    print(EntropySampling(models, loader))

