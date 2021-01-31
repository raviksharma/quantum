from math import sqrt
import numpy as np

from base import zero, one, \
                 zz, zo, oz, oo
from gates import I, H, K, CNOT

#  entangled quantum state
#
#                +---+
#        |0> ----+ H +-----.---         |00> + |11>
#                +---+     |      }     -----------    
#                          |              sqrt(2)
#        |0> --------------O---
#

def test_bell_state():
    assert(np.allclose(CNOT(K(zz)), (zz + oo)/sqrt(2)))
