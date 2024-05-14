import pytest
from dooc import datasets


def test_MutSmi():
    ds = datasets.MutSmi()
    return
    
def test_MutSmiXAttention():
    ds = datasets.MutSmiXAttention()
    entry = ds(["CC[N+](C)(C)Cc1ccccc1Br"], [[1, 0, 1]], [0.85])
    return

def test_MutSmiFullConnection():
    ds = datasets.MutSmiFullConnection()
    entry = ds(["CC[N+](C)(C)Cc1ccccc1Br"], [[1, 0, 1]], [0.85])
    return
