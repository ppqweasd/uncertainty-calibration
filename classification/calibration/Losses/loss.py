"""
Implementation of the following loss functions:
1. Cross Entropy
2. Focal Loss
3. Cross Entropy + MMCE_weighted
4. Cross Entropy + MMCE
5. Brier Score
6. DECE
"""

from torch.nn import functional as F
from Losses.focal_loss import FocalLoss
from Losses.focal_loss_adaptive_gamma import FocalLossAdaptive
from Losses.mmce import MMCE, MMCE_weighted
from Losses.brier_score import BrierScore
from Losses.dece import DECE
import torch


def cross_entropy(logits, targets, **kwargs):


    return F.cross_entropy(logits, targets, reduction="sum")


def focal_loss(logits, targets, **kwargs):
    return FocalLoss(gamma=kwargs["gamma"])(logits, targets)


def focal_loss_adaptive(logits, targets, **kwargs):
    return FocalLossAdaptive(gamma=kwargs["gamma"], device=kwargs["device"])(
        logits, targets
    )


def mmce(logits, targets, **kwargs):
    ce = F.cross_entropy(logits, targets)
    mmce = MMCE(kwargs["device"])(logits, targets)
    return ce + (kwargs["lamda"] * mmce)


def mmce_weighted(logits, targets, **kwargs):
    ce = F.cross_entropy(logits, targets)
    mmce = MMCE_weighted(kwargs["device"])(logits, targets)
    return ce + (kwargs["lamda"] * mmce)


def brier_score(logits, targets, **kwargs):
    return BrierScore()(logits, targets)


def dece(logits, targets, **kwargs):
    return DECE(kwargs["device"], kwargs["num_bins"], kwargs["t_a"], kwargs["t_b"])(
        logits, targets
    )



def entropy2(outputs,labels):
    res = 0
    num=0.0001

    # print(data)

    for i, element in enumerate(outputs):
        #print(element,labels[i,:])
        _, predicted = torch.max(torch.sigmoid(element), dim=0)
        _, label = torch.max(labels[i,:], dim=0)
       # print(predicted,label)

        if predicted == label:

            a = torch.max(torch.sigmoid(element), dim=-1)
        #print(a.values)

            res += a.values
            num+=1
            #print('#################')
            #print(res)

    #print(res.cpu().detach().numpy().shape)
    return res / num


def map(outputs,labels):
    total=0
    correct=0

    _, predicted = torch.max(torch.sigmoid(outputs.data), 1)
    _, label= torch.max(labels.data, 1)
    #print(predicted, label)
    #print(predicted)
    total += labels.size(0)
    correct += (predicted == label).sum().item()
    #print(correct)
    #print(correct.cpu().detach().numpy().shape)
    return correct / total

def class_label(logits, targets, **kwargs):
    criterion = torch.nn.BCEWithLogitsLoss()
    #print(logits)
    #print(targets)

    if map(logits, targets) > 0.6:

        loss = criterion(logits, targets * (1 +  (map(logits, targets) - entropy2(logits.data, targets.data))))#+FocalLoss(gamma=kwargs["gamma"])(logits, targets)
    else:
        loss = criterion(logits, targets)#+FocalLoss(gamma=kwargs["gamma"])(logits, targets)

    return loss

    #loss += self.nn_kl_divergence() * complexity_cost_weight