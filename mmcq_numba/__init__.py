#! -*- coding: utf-8 -*-
from .quantize import mmcq
from numba import jit

__version__ = '0.1.2'
__all__ = '__version__', 'get_dominant_color', 'get_palette', 'mmcq'
