import cupy
import numpy


class CP:
    cp = numpy
    is_gpu_used = False

    def use_cpu():
        CP.cp = numpy
        CP.is_gpu_used = False

    def use_gpu():
        CP.cp = cupy
        CP.is_gpu_used = True

    def copy(obj):
        if CP.is_gpu_used:
            return cupy.asarray(obj)
        return numpy.copy(obj)

    def asnumpy(obj):
        if CP.is_gpu_used:
            return cupy.asnumpy(obj)
        return numpy.asarray(obj)
