import torch
import pytest
from dooc import nets


def test_GNN():
    model = nets.GNN()
    model.forward([1, 0, 0, 1])
    return
