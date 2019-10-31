import pytest
from workshop.train import hello


def test_hello():
    hello()
    assert True
