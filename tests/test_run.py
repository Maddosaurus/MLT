# pylint: disable=redefined-outer-name,missing-docstring,unused-import,no-self-use
import pytest
import argparse

from .context import run

def test_create_parser():
    assert isinstance(run.create_parser(), argparse.ArgumentParser)