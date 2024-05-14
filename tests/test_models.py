import torch
from dooc import models
from torch import nn


def test_MutSmi():
    cls = models.MutSmi()
    return

def test_MutSmiXAttention():
    model = models.MutSmiXAttention()
    assert model.forward([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]) == None
    return

def test_MutSmiFullConnection():
    model = models.MutSmiFullConnection()
    assert model.forward([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]) == None
    return
