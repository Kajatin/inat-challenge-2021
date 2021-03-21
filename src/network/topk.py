import torch
from torch.nn.functional import softmax


def topk(outputs, targets, k=1):
    predictions = torch.topk(softmax(outputs, 1), k, 1).indices
    predictions = predictions.t()
    correct = predictions.eq(targets.view(1, -1).expand_as(predictions))
    error = targets.shape[0] - correct.flatten().float().sum()
    return error
