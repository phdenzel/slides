#!/usr/bin/env python3
"""
Animation objects for astronomical bodies
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anima
from animobj import AnimObj


class AstroObj(AnimObj):
    """
    Base class for all astro animations
    """
    def __init__(self, suffix: (int, str) = None, **kwargs):
        AstroObj.parse_config(self, kwargs)
        if suffix:
            if isinstance(suffix, int):
                suffix = str(suffix)
            basename = "_".join([self.basename, str(suffix)])
        else:
            basename = self.basename if hasattr(self, 'basename') else None
        AnimObj.__init__(self, basename, **kwargs)


class Photon(AstroObj):
    CONFIG = dict(
        basename="photon",
        flip=False,
        # stypes=[str(nmbr) for nmbr in range(1, 10)],
    )


class Galaxy(AstroObj):
    CONFIG = dict(
        flip=False,
        basename="galaxy",
        # stypes=[str(nmbr) for nmbr in range(1, 10)],
    )


class RadioDish(AstroObj):
    CONFIG = dict(
        flip=False,
        basename="radio_dish"
        # stypes=[str(nmbr) for nmbr in range(1, 10)]
    )

    
class SignalResponse(AstroObj):
    CONFIG = dict()
