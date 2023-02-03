#! -*- coding: utf-8 -*-
from collections import Iterator

from .math_ import euclidean


__all__ = 'CMap', 'PQueue',

from .color import get_color_index
from .constant import RSHIFT, SIGBITS
from numba import jit, f8, i8, b1, void
from numba.types import Tuple

@jit(Tuple((i8, i8, i8))(i8, i8, i8,i8, i8, i8, i8[:]),nopython=True, cache=True)
def average_numba(r1,r2,g1,g2,b1,b2,histo):
    total = 0
    mult = 1 << (8 - SIGBITS)
    r_sum = 0
    g_sum = 0
    b_sum = 0
    for i in range(r1, r2 + 1):
        for j in range(g1, g2 + 1):
            for k in range(b1, b2 + 1):
                index = get_color_index(i, j, k)
                hval = histo[index]
                total += hval
                r_sum += hval * (i + 0.5) * mult
                g_sum += hval * (j + 0.5) * mult
                b_sum += hval * (k + 0.5) * mult

    if total:
        r_avg = ~~int(r_sum / total)
        g_avg = ~~int(g_sum / total)
        b_avg = ~~int(b_sum / total)
    else:
        r_avg = ~~int(mult * (r1 + r2 + 1) / 2)
        g_avg = ~~int(mult * (g1 + g2 + 1) / 2)
        b_avg = ~~int(mult * (b1 + b2 + 1) / 2)

    return r_avg, g_avg, b_avg

class CMap:

    def __init__(self):
        self.vboxes = []

    @property
    def palette(self):
        return [d['color'] for d in self.vboxes]

    def append(self, item):
        avg = item._avg
        avg = average_numba(item.r1,item.r2,item.g1,item.g2,item.b1,item.b2,item.histo) if avg is None else avg
        self.vboxes.append({'vbox': item, 'color': avg})

    def __len__(self):
        return len(self.vboxes)

    def nearest(self, color):
        if not self.vboxes:
            raise ValueError('Empty VBoxes!')

        min_d = float('Inf')
        p_color = None
        for vbox in self.vboxes:
            vbox_color = vbox.color
            distance = euclidean(color, vbox_color)
            if min_d > distance:
                min_d = distance
                p_color = vbox.color

        return p_color

    def map(self, color):
        for vbox in self.vboxes:
            if vbox.contains(color):
                return vbox.color

        return self.nearest(color)

import time

class PQueue(Iterator):

    def __init__(self, sorted_key):
        self.sorted_key = sorted_key
        self.items = []

    def __next__(self):
        if not self.items:
            raise StopIteration()

        return self.pop()

    def append(self, item):
        self.items.append(item)
        
    def sort(self):
        #self.items.sort(key=self.sorted_key, reverse=True)
        self.items = sorted(self.items, key=self.sorted_key, reverse=True)

    def pop(self):
        start = time.time()
        #self.items = sorted(self.items, key=self.sorted_key, reverse=False)
        #if len(self.items) > 1:
        self.sort()
        #print("sorted",len(self.items), time.time()-start)
        start = time.time()
        popout = self.items.pop(0)
        #print("pop",len(self.items), time.time()-start)
        return popout

    def __len__(self):
        return len(self.items)
