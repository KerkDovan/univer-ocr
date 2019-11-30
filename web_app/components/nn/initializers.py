from .gpu import CP


def xavier_normal(in_num, out_num):
    a = 1 / CP.cp.sqrt(in_num)
    w = a * CP.cp.random.normal(size=(in_num, out_num))
    return w


def xavier_uniform(in_num, out_num):
    a = 1 / CP.cp.sqrt(in_num)
    w = a * CP.cp.random.uniform(size=(in_num, out_num))
    return w


def kaiming_normal(in_num, out_num):
    a = 1 / CP.cp.sqrt(in_num / 2)
    w = a * CP.cp.random.normal(size=(in_num, out_num))
    return w


def kaiming_uniform(in_num, out_num):
    a = 1 / CP.cp.sqrt(in_num / 2)
    w = a * CP.cp.random.uniform(size=(in_num, out_num))
    return w
