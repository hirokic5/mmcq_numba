from .constant import SIGBITS
from numba import jit, f8, i8, b1, void

@jit(i8(i8,i8,i8),nopython=True, cache=True)
def get_color_index(r, g, b):
    return (r << (2 * SIGBITS)) + (g << SIGBITS) + b
