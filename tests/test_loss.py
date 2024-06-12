import torch
from dooc import loss

def test_ListNetLoss():
    predict = torch.randn(5, 3)
    target = torch.randn(5, 3)
    loss_mean = loss.ListNetLoss(reduction='mean')
    mean = loss_mean(predict, target)
    loss_sum = loss.ListNetLoss(reduction='sum')
    sum = loss_sum(predict, target)
    assert sum / 15 == mean
