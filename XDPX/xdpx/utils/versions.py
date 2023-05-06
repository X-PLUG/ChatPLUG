import torch
from packaging import version


def torch_ge_120():
    return version.parse(torch.__version__) >= version.parse('1.2')


def torch_lt_120():
    return version.parse(torch.__version__) < version.parse('1.2')


def torch_ne_13():
    return version.parse(torch.__version__).release[:2] != (1, 3)


def torch_ge_150():
    return version.parse(torch.__version__) >= version.parse('1.5')