#!/usr/bin/env python3
"""
Animate the propagation of objects for presentations
"""
import os
import itertools as it
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image
import cv2


def compile_config(obj: object, kwargs: dict, caller_locals: dict = {}):
    """
    Sets a class' init args and CONFIG values as local variables

    Args:
        obj (instance): class instance
        kwargs (dict): keywords directly passed to the class' __init__
        caller_locals (dict): additional locals to set

    Returns:
        None
    """
    classes_in_hierarchy = [obj.__class__]
    static_configs = []
    while len(classes_in_hierarchy) > 0:
        cls = classes_in_hierarchy.pop()
        classes_in_hierarchy += cls.__bases__
        if hasattr(cls, "CONFIG"):
            static_configs.append(cls.CONFIG)
    # sanitize
    locs = caller_locals.copy()
    for arg in ["self", "kwargs"]:
        locs.pop(arg, caller_locals)
    all_dcts = [kwargs, locs, obj.__dict__]
    all_dcts += static_configs
    obj.__dict__ = recurse_merge(*reversed(all_dcts))


def recurse_merge(*dicts):
    """
    Recursive merge multiple dictionaries
    Entries of later dictionaries override earier ones

    Args:
        dicts (*tuple[dict]): iteration of dictionaries to merge

    Returns:
        (dict): single, fully merged dictionary
    """
    dct = dict()
    all_items = it.chain(*[d.items() for d in dicts])
    for key, val in all_items:
        if key in dct and isinstance(dct[key], dict) and isinstance(val, dict):
            dct[key] = recurse_merge(dct[key], val)
        else:
            dct[key] = val
    return dct


class AnimObj(object):
    CONFIG = dict(
        image_mode="RGBA",
        size=1.,
        scale=1.,
        flip=True,
        alignment=0.,
    )

    DEG2RAD = np.pi/180
    SQRT2 = np.sqrt(2)

    def __init__(self, basename: str,
                 ext: str = ".png",
                 imgdir: str = "resources/",
                 position: (complex, tuple, list, np.ndarray) = 0j,
                 rotation: (float) = 0,
                 **kwargs):
        """
        Args:
            basename (str): name of an image object
            ext (str): filename extension
            imgdir (str): path to the image to be animated
            position (complex/tuple|list|np.ndarray): position of the image
            rotation (float): rotation angle in degrees
            kwargs (dict): additional configs
        """
        AnimObj.parse_config(self, kwargs)
        self.filename = os.path.join(imgdir, basename+ext) if basename else ""
        self.steps = 0
        self.position = position
        if isinstance(position, complex):
            self.position = position
        elif isinstance(position, (tuple, list, np.ndarray)):
            self.position = position[0] + 1j*position[1]
        if self.filename and os.path.exists(self.filename):
            image = Image.open(self.filename).convert(self.image_mode)
            self.pixel_array = np.array(image)
            if self.flip:
                self.pixel_array = np.fliplr(self.pixel_array)
            self.image = image
        else:
            image = None
            self.pixel_array = None
        if rotation > 0:
            self.rotate(rotation)

    @staticmethod
    def parse_config(obj, kwargs: dict, caller_locals: dict = {}):
        """
        Wrapper for `compile_config`
        """
        compile_config(obj, kwargs, caller_locals=caller_locals)

    def reset(self, position, phase_io=None, size=1., scale=1., rotation=None):
        """
        Reset the object

        Args:
            position (complex/tuple|list|np.ndarray): position of the image
            phase_io (float): rotation angle in degrees
        """
        self.steps = 0
        self.alignment = 0
        self.size = size
        self.scale = scale
        self.pixel_array = np.array(self.image)
        if self.flip:
            self.pixel_array = np.fliplr(self.pixel_array)
        if isinstance(position, complex):
            self.position = position
            theta = np.arctan2(self.position.imag, self.position.real) \
                / AnimObj.DEG2RAD
        elif isinstance(position, (int, float)):
            self.position = position + 0j
            theta = 0
        elif isinstance(position, (tuple, list, np.ndarray)):
            self.position = position[0] + position[1]*1j
            theta = np.arctan2(self.position.imag, self.position.real) \
                / AnimObj.DEG2RAD
        if rotation:
            self.rotate(rotation)
        elif theta != 0:
            self.rotate(theta, inplace=True)
        if phase_io:
            self.phase_io(phase_io)

    def increment(self):
        self.steps += 1

    @property
    def aspect(self) -> float:
        if self.pixel_array is not None:
            h, w, _ = self.pixel_array.shape
        elif hasattr(self, 'graph_dim'):
            h, w = self.graph_dim['v'], 2*self.graph_dim['h']
        else:
            h = w = self.size
        return h / w

    @property
    def extent(self) -> list:
        x, y = self.position.real, self.position.imag
        w2 = 0.5 * self.size * self.scale
        h2 = 0.5 * self.size * self.scale * self.aspect
        l, r, b, t = x-w2, x+w2, y-h2, y+h2
        xtnt = [l, r, b, t]
        return xtnt

    def max_extent(self, *extents):
        xtnt = self.extent
        for e in extents:
            if e[0] < xtnt[0]:
                xtnt[0] = e[0]
            if e[1] > xtnt[1]:
                xtnt[1] = e[1]
            if e[2] < xtnt[2]:
                xtnt[2] = e[2]
            if e[3] > xtnt[3]:
                xtnt[3] = e[3]
        return xtnt

    @property
    def bounding_radius(self) -> float:
        l, r, b, t = self.extent
        dim = abs(l-r)**2 + abs(b-t)**2
        return np.sqrt(dim)/2

    @property
    def distance(self) -> float:
        return abs(self.position)

    def move(self, delta,
             radial: bool = False,
             limit: float = 0,
             increment: bool = True):
        """
        Move the center position of objects

        Args:
            delta (int|float/complex/tuple|list|np.ndarray): translation added to position
            radial (bool): radially move the position to the center
            limit (float): a maximum value for the radial position (no effect if radial=False)
            increment (bool): if True, the object's step increments by 1
        """
        if radial:
            if isinstance(delta, (int, float)):
                r = abs(self.position) - delta*AnimObj.SQRT2
                theta = np.arctan2(self.position.imag, self.position.real)
                r = max(r, limit)
                self.position = r * np.e**(1j*theta)
            elif isinstance(delta, complex):
                r = abs(self.position + delta)
                theta = np.arctan2(self.position.imag, self.position.real)
                r = max(r, limit)
                self.position = r * np.e**(1j*theta)
            elif isinstance(delta, (tuple, list, np.ndarray)):
                r = abs(self.position + complex(delta[0], delta[1]))
                theta = np.arctan2(self.position.imag, self.position.real)
                r = max(r, limit)
                self.position = r * np.e**(1j*theta)
        else:
            if isinstance(delta, (int, float)):
                self.position += delta + 1j*delta
            elif isinstance(delta, complex):
                self.position += delta
            elif isinstance(delta, (tuple, list, np.ndarray)):
                self.position.real += delta[0]
                self.position.imag += delta[1]
        if increment:
            self.increment()

    def rotate(self, theta: float, inplace: bool = True):
        """
        Rotate the object by a given angle
        
        Args:
            theta (float): rotation angle in degrees
            inplace (bool): if True the rotation is about the object's position
        """
        self.alignment = (self.alignment + theta) % 360
        self.pixel_array = ndimage.rotate(self.pixel_array, theta, reshape=True)
        if not inplace:
            self.position = self.position * np.e**(1j*theta*AnimObj.DEG2RAD)

    def phase_io(self, ratio: float):
        """
        Phase objects in or out by alpha channel multiplication

        Args:
            ratio (float): relative phase ratio
        """
        alpha = self.pixel_array[:, :, 3]
        factor = 1. + ratio
        self.pixel_array[:, :, 3] = cv2.multiply(alpha, factor)

    def phase_shift(self, ratio: float):
        """
        Phase objects in or out by alpha channel multiplication

        Args:
            ratio (float): relative phase ratio
        """
        alpha = self.pixel_array[:, :, 3]
        msk = alpha > 0
        if self.pixel_array.dtype == np.uint8:
            shift = np.uint8(max(1, abs(ratio)*255))
        else:
            shift = abs(ratio)
        if ratio > 0:
            self.pixel_array[:, :, 3] = cv2.add(self.pixel_array[:, :, 3], msk*shift)
        else:
            self.pixel_array[:, :, 3] = cv2.subtract(self.pixel_array[:, :, 3], msk*shift)

    def rescale(self, factor: float):
        """
        Rescale objects by a given factor

        Args:
            factor (float): scaling factor
        """
        self.scale *= factor

    def add_arc(self,
                arc_range: (tuple, list) = [0, np.pi/2],
                arc_N: int = 200,
                c_offset: complex = 0,
                r_offset: float = 0,
                **kwargs):
        """
        Plot circular arc at object's position with given angular range
        
        Args:
            arc_range (tuple, list): angle limits for arc
            c_offset (complex): center offset of the arc
            r_offset (float): radial offset of the arc

        Returns:
            (matplotlib.lines.Line2D): a Line2D instance
        """
        kwargs.setdefault('color', 'white')
        kwargs.setdefault('lw', 3)
        kwargs.setdefault('alpha', 1)
        self.arc_range = arc_range
        self.arc_N = arc_N
        self.arc_offset = {'center': c_offset,
                           'radius': r_offset}
        self.arc_kwargs = kwargs
        arc_xs, arc_ys = self.arc
        ax = plt.gca()
        return ax.plot(arc_xs, arc_ys, **kwargs)[0]

    @property
    def arc(self):
        if hasattr(self, 'arc_range'):
            arc_range = self.arc_range
        else:
            arc_range = [0, np.pi/2]
        if hasattr(self, 'arc_N'):
            arc_N = self.arc_N
        else:
            arc_N = 0
        if hasattr(self, 'arc_offset'):
            c_offset = self.arc_offset['center']
            r_offset = self.arc_offset['radius']
        else:
            c_offset, r_offset = complex(0, 0), 0
        arc_angles = np.linspace(arc_range[0], arc_range[1], arc_N)
        r = abs(self.position) + r_offset
        arc_xs = c_offset.real + r * np.cos(arc_angles)
        arc_ys = c_offset.imag + r * np.sin(arc_angles)
        return arc_xs, arc_ys

    def add_graph(self,
                  signal: float = 0,
                  gain: float = 1,
                  sigma: float = 1,
                  func: callable = None,
                  flat: bool = True,
                  v: float = None,
                  h: float = 1,
                  N: int = 200,
                  **kwargs):
        """
        Plot graph like lines at position with given vertical and horizontal sizes

        Args:
            signal (float):
            gain (float):
            sigma (float):
            func (callable):
            flat (bool):
            v (float):
            h (float):
            N (int):
        """
        kwargs.setdefault('color', 'white')
        kwargs.setdefault('lw', 3)
        kwargs.setdefault('alpha', 1)
        self.graph_kwargs = kwargs
        self.graph_N = N
        self.graph_dim = {'v': v, 'h': h}
        if self.pixel_array is None:
            self.size = 2*h
        self.graph_signal = signal
        self.graph_gain = gain
        self.graph_sigma = sigma
        self.graph_flat = lambda x: np.ones_like(x)*(self.position.imag)
        self.graph_func = func if func \
            else lambda x: self.position.imag  \
            + 2*np.pi/3 * self.graph_gain \
            * (x - self.graph_signal)/self.graph_sigma \
            * np.exp(-(x - self.graph_signal)**2 / self.graph_sigma**2)
        ax = plt.gca()
        graph_xs, graph_ys = self.graph(flat=flat)
        return ax.plot(graph_xs, graph_ys, **kwargs)[0]

    def graph(self, flat=False, signal_inc=0, gain_inc=0):
        self.graph_signal += signal_inc
        self.graph_gain += gain_inc
        graph_xs = []
        graph_ys = []
        o = self.position
        N = self.graph_N
        v, h = self.graph_dim['v'], self.graph_dim['h']
        if v:
            graph_xs = [o.real, o.real, o.real]
            graph_ys = [o.imag-v/2, o.imag+v/2, o.imag]
        x = np.linspace(o.real, o.real+h, N)
        if flat:
            y = self.graph_flat(x)
        else:
            y = self.graph_func(x)
        graph_xs += list(x)
        graph_ys += list(y)
        return graph_xs, graph_ys
