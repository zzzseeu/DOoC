import os.path
import pytest


@pytest.fixture
def datadir():
    return os.path.join(os.path.dirname(__file__), '../dooc/data')
