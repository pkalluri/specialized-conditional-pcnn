import torch
import torch.nn.functional as F

class OfficialLossModule(torch.nn.Module): # define as Module so e.g. can call cuda later
    def __init__(self):
        super(OfficialLossModule, self).__init__()
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.nll = torch.nn.NLLLoss(reduction='mean')
    def forward(self, predictions, targets): # probs: (batchsize, num_possible_values, W, H), target: (batchsize, W, H)
        loss = self.nll(self.logsoftmax(predictions), targets)
        return loss
official_loss_function = OfficialLossModule()

class StandardLossModule(torch.nn.Module): # define as Module so e.g. can call cuda later
    def __init__(self):
        super(StandardLossModule, self).__init__()
    def forward(self, predictions, targets): # probs: (batchsize, num_possible_values, W, H), target: (batchsize, W, H)
        Y_loss = F.cross_entropy(predictions, targets, reduction='mean') # sum in log space (all y in Y)
        return Y_loss
standard_loss_function = StandardLossModule()

class SumLossModule(torch.nn.Module):
    def __init__(self):
        super(SumLossModule, self).__init__()

    def forward(self, predictions, targets): # probs: (batchsize, num_possible_values, W, H), each target: (batchsize, 1, W, H)
        y_losses = F.cross_entropy(predictions, targets, reduction='none')
        y_losses = torch.sum(y_losses,dim=[1,2]) # sum in log space (all pixels)
        Y_loss = torch.logsumexp(y_losses, dim=0) # sum in non log-space (any y in Y)
        return Y_loss
sum_loss_function = SumLossModule()

class MinLossModule(torch.nn.Module):
    def __init__(self):
        super(MinLossModule, self).__init__()

    def forward(self, predictions, targets): # probs: (batchsize, num_possible_values, W, H), each target: (batchsize, 1, W, H)
        y_losses = F.cross_entropy(predictions, targets, reduction='none')
        y_losses = torch.sum(y_losses, dim=[1,2])  # sum in log space (all pixels)
        Y_loss = torch.min(y_losses)  # max (any y in Y)
        return Y_loss
min_loss_function = MinLossModule()