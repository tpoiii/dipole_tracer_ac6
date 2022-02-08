"""
tictoc
provides simple tic, toc interface to time code blocks
"""

from __future__ import print_function
import time

_tic = time.time()

def tic():
    """start = tic()
    start tictoc timer
    start - time in seconds
    """
    global _tic
    _tic = time.time()
    return _tic

def toc(quiet=False,start=None):
    """elapsed = toc(quiet=False,start=None)
    print elapsed time since call to tic()
    quiet - suppress print output
    start - use alternate start time
    elapsed - seconds elapsed
    """
    if start is None:
        start = _tic
    elapsed = time.time()-start
    if not quiet:
        print('%g seconds elapsed' % (elapsed))
    return elapsed

