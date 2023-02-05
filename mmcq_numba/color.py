from numba import i8, jit

from .constant import SIGBITS


@jit(i8(i8, i8, i8), nopython=True, cache=True)
def get_color_index(r, g, b):
    return (r << (2 * SIGBITS)) + (g << SIGBITS) + b
