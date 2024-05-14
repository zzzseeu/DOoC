import torch
import pytest
from dooc import pipelines

def test_MutSmi():
    pipeline = pipelines.MutSmi()
    return

def test_MutSmiXAttention():
    pipeline = pipelines.MutSmiXAttention()
    return pipeline([1, 0, 0, 1], "CC[N+](C)(C)Cc1ccccc1Br")

def test_MutSmiFullConnection():
    pipeline = pipelines.MutSmiFullConnection()
    return pipeline([1, 0, 0, 1], "CC[N+](C)(C)Cc1ccccc1Br")
