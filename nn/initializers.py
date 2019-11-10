import numpy as np


def xavier_normal(in_num, out_num):
    a = 1 / np.sqrt(in_num)
    w = a * np.random.normal(size=(in_num, out_num))
    return w


def xavier_uniform(in_num, out_num):
    a = 1 / np.sqrt(in_num)
    w = a * np.random.uniform(size=(in_num, out_num))
    return w


def kaiming_normal(in_num, out_num):
    a = 1 / np.sqrt(in_num / 2)
    w = a * np.random.normal(size=(in_num, out_num))
    return w


def kaiming_uniform(in_num, out_num):
    a = 1 / np.sqrt(in_num / 2)
    w = a * np.random.uniform(size=(in_num, out_num))
    return w
