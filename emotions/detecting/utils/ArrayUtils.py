import random


def divide_into_two_arrays(pair_array):
    x_arr = []
    y_arr = []
    for (x, y) in pair_array:
        x_arr += [x]
        y_arr += [y]
    return x_arr, y_arr


def shuffle(array):
    random.shuffle(array)


def cut_left(array, count):
    count = max(count, 1)
    return array[:count]


def cut_right(array, count):
    count = max(count, 1)
    return array[-count:]


def cut(array, _from, _to):
    _from = max(_from, 1)
    _to = max(_to, 1)
    return array[_from:_to]


def last(array):
    return array[-1]
