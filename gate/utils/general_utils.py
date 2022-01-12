import torch


def compute_accuracy(inputs, targets):
    acc = targets == inputs.argmax(-1)
    return torch.mean(acc.type(torch.float32)) * 100.0
