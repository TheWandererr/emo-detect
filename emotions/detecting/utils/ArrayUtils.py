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