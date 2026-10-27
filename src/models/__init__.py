from .BasicModel import BasicModel
from .rirci import RIRCI


def basic(**kwargs):
    return BasicModel(**kwargs)


def rirci(**kwargs):
    return RIRCI(**kwargs)
