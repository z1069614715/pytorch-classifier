import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ['PolyLoss', 'CrossEntropyLoss', 'FocalLoss', 'RDropLoss']

class PolyLoss(torch.nn.Module):
    """
    PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions
    <https://arxiv.org/abs/2204.12511>
    """
    def __init__(self, label_smoothing: float = 0.0, weight: torch.Tensor = None, epsilon=2.0):
        super().__init__()
        self.epsilon = epsilon
        self.label_smoothing = label_smoothing
        self.weight = weight

    def forward(self, outputs, targets):
        ce = F.cross_entropy(outputs, targets, label_smoothing=self.label_smoothing, weight=self.weight)
        pt = F.one_hot(targets, outputs.size()[1]) * F.softmax(outputs, 1)

        return (ce + self.epsilon * (1.0 - pt.sum(dim=1))).mean()

class CrossEntropyLoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.0, weight: torch.Tensor = None):
        super(CrossEntropyLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.cross_entropy(input, target)

class FocalLoss(nn.Module):
    def __init__(self, label_smoothing:float = 0.0, weight: torch.Tensor = None, gamma:float = 2.0):
        super(FocalLoss, self).__init__()
        self.label_smoothing = label_smoothing
        self.weight = weight
        self.gamma = gamma
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_onehot = F.one_hot(target, num_classes=input.size(1))
        target_onehot_labelsmoothing = torch.clamp(target_onehot.float(), min=self.label_smoothing/(input.size(1)-1), max=1.0-self.label_smoothing)
        input_softmax = F.softmax(input, dim=1) + 1e-7
        input_logsoftmax = torch.log(input_softmax)
        ce = -1 * input_logsoftmax * target_onehot_labelsmoothing
        fl = torch.pow((1 - input_softmax), self.gamma) * ce
        fl = fl.sum(1) * self.weight[target.long()]
        return fl.mean()

class RDropLoss(nn.Module):
    def __init__(self, loss, a=0.3):
        super(RDropLoss, self).__init__()
        self.loss = loss
        self.a = a

    def forward(self, input, target: torch.Tensor) -> torch.Tensor:
        if type(input) is list:
            input1, input2 = input
            main_loss = (self.loss(input1, target) + self.loss(input2, target)) * 0.5
            kl_loss = self.compute_kl_loss(input1, input2)
            return main_loss + self.a * kl_loss
        else:
            return self.loss(input, target)
    
    def compute_kl_loss(self, p, q, pad_mask=None):
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
        
        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)

        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss