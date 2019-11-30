import cupy
import numpy


class CP:
    cp = numpy
    use_gpu = False

    def use_cpu():
        CP.cp = numpy
        CP.use_gpu = False

    def use_gpu():
        CP.cp = cupy
        CP.use_gpu = True
