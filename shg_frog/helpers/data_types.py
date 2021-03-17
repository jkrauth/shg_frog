""" Data types defined for shg_frog """

from typing import NamedTuple
import numpy as np

class Data(NamedTuple):
    """ Data structure to contain frog image and meta data """
    image: np.ndarray
    meta: dict
