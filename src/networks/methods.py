from src.networks.rirci import RIRCI


def rirci(**kwargs):
    return RIRCI(args=kwargs['args'], shared_depth=1, blocks=2, long_skip=True)
