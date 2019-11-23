from collections import Iterable


def make_list_if_not(var):
    if isinstance(var, list):
        return var
    return [var]


def tuplize(name, var, length):
    is_negative = False
    result = None

    if isinstance(var, int):
        is_negative = var < 0
        result = tuple(var for _ in range(length))

    elif isinstance(var, Iterable):
        tmp = tuple(var)
        if len(tmp) == length and all(isinstance(x, int) for x in tmp):
            is_negative = any(x < 0 for x in tmp)
            result = tmp

    if is_negative:
        raise ValueError(f'{name} cannot be negative, found: {var}')
    if result is None:
        raise TypeError(
            f'{name} must be either int or iterable of ints of length {length}, '
            f'found {type(var).__name__}')

    return result
