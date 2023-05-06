from multiprocessing.managers import SyncManager
import os
import pdb
import sys

__all__ = ['set_trace']
manager = SyncManager()
manager.start()
registry = manager.dict()
barrier = manager.Barrier(1)


def set_nprocs(n):
    global barrier
    barrier = manager.Barrier(n)


class ForkedPdb(pdb.Pdb):
    """
    A Pdb subclass that may be used from a forked multiprocessing child
    """

    def interaction(self, *args, **kwargs):
        global registry, barrier
        if barrier.parties > 1:
            pid = os.getpid()
            if pid not in registry:
                registry[pid] = False
            barrier.wait()
            master_pid = min(registry.keys())
            is_master = pid == master_pid
        else:
            is_master = True

        if is_master:
            _stdin = sys.stdin
            try:
                sys.stdin = open('/dev/stdin')
                pdb.Pdb.interaction(self, *args, **kwargs)
            finally:
                sys.stdin = _stdin
        
        if barrier.parties > 1:
            barrier.wait()


def set_trace():
    pdb = ForkedPdb()
    pdb.set_trace(sys._getframe().f_back)
