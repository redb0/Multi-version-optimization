import numpy as np


def get_function(g, nuclear_func_index, selectivity_factor):

    # ядро
    # линейное
    if nuclear_func_index == 1:
        return (1 - g)**selectivity_factor

    # параболическое
    if nuclear_func_index == 2:
        return (1 - g**2)**selectivity_factor

    # кубическо
    if nuclear_func_index == 3:
        return (1 - g**3)**selectivity_factor

    # экспоненциальное
    if nuclear_func_index == 4:
        return np.exp(-selectivity_factor * g)

    # гипербалическое
    # if nuclear_func_index == 5:
    #     return g ** (-selectivity_factor)
