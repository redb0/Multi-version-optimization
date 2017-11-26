import numpy as np


def get_range(f_index):
    dim = 30

    if f_index == 1:
        low = -100
        up = 100
        return low, up, dim

    if f_index == 2:
        low = -50
        up = 50
        dim = 2
        return low, up, dim

    if f_index == 3:
        low = -50
        up = 50
        dim = 2
        return low, up, dim

    if f_index == 4:
        low = -10
        up = 10
        dim = 2
        return low, up, dim

    if f_index == 5:
        low = -6
        up = 6
        dim = 2
        return low, up, dim

    if f_index == 6:
        low = -30
        up = 30
        dim = 30
        return low, up, dim

    if f_index == 7:
        low = -5.12
        up = 5.12
        dim = 2
        return low, up, dim

    if f_index == 8:
        low = -1.28
        up = 1.28
        dim = 30
        return low, up, dim

    if f_index == 9:
        low = -32
        up = 32
        return low, up, dim

    if f_index == 10:
        low = -65.536
        up = 65.536
        # low = -40
        # up = 40
        dim = 2
        return low, up, dim

    if f_index == 11:
        low = -6
        up = 6
        dim = 2
        return low, up, dim

    if f_index == 12:
        low = 0
        up = 10
        dim = 2
        return low, up, dim

    if f_index == 13:
        low = -10
        up = 10
        dim = 2
        return low, up, dim

    if f_index == 14:
        low = -10
        up = 10
        dim = 2
        return low, up, dim

    if f_index == 15:
        low = -10
        up = 10
        dim = 2
        return low, up, dim

    if f_index == 16:
        low = -6
        up = 6
        dim = 2
        return low, up, dim

    if f_index == 17:
        low = np.array([-5, 0])
        up = np.array([10, 15])
        dim = 2
        return low, up, dim

    if f_index == 18:
        low = -5
        up = 5
        dim = 2
        return low, up, dim

    if f_index == 19:
        low = -6
        up = 6
        dim = 2
        return low, up, dim
