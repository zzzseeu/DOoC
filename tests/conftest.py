import os.path
import pytest
from moltx import tokenizers as tkz
from moltx.models import AdaMRTokenizerConfig


@pytest.fixture
def datadir():
    return os.path.join(os.path.dirname(__file__), '../dooc/data')


@pytest.fixture
def smi_tkz():
    return tkz.MoltxTokenizer.from_pretrain(
        conf=AdaMRTokenizerConfig.Prediction
    )
