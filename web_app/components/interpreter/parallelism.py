import multiprocessing
from multiprocessing import Process
from multiprocessing.managers import RemoteError
from multiprocessing.pool import Pool as ProcessPool
from multiprocessing.pool import ThreadPool
from threading import Thread

ERRORS_TO_STOP = (KeyboardInterrupt, BrokenPipeError, EOFError, RemoteError)


class MP:
    mp = multiprocessing
    Pool = ThreadPool
    Process = Thread

    is_multiprocessing_used = False

    @staticmethod
    def use_multiprocessing():
        MP.Pool = ProcessPool
        MP.Process = Process
        MP.is_multiprocessing_used = True

    @staticmethod
    def use_threading():
        MP.Pool = ThreadPool
        MP.Process = Thread
        MP.is_multiprocessing_used = False
