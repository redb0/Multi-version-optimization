import numpy as np
from functools import reduce


def test_function(x_i, f_index, dim):
    if f_index == 1:
        """Функция сферы. Одноэкстремальная. Глобальный минимум = 0 в точке: [0, 0]"""
        return np.sum(x_i ** 2)

    if f_index == 2:
        # истинный минимум в [0 0]
        return 3 * x_i[0]**2 + 4 * x_i[0] * x_i[1] + 5 * x_i[1]**2

    if f_index == 3:
        # sin(((x - 2) ** 2 + (y - 5) ** 2)) * (x - 2) ** 2 + (y - 5) ** 2
        return (np.sin((x_i[0]-2)**2 + (x_i[1]-5)**2)) * ((x_i[0]-2)**2) + (x_i[1]-5)**2

    if f_index == 4:
        # истинный минимум в районе от [3.4 3.4] до [4 4]
        # -((exp((-(((x - 4) ^ 2 + (y - 4) ^ 2) ^ 2)) / 1000) + exp((-(((x + 4) ^ 2 + (y + 4) ^ 2) ^ 2)) / 1000) + 0.15
        # exp(-(((x + 4) ^ 2 + (y + 4) ^ 2) ^ 2)) + 0.15
        # exp(-(((x - 4) ^ 2 + (y - 4) ^ 2) ^ 2))))
        a = np.exp((-(((x_i[0] - 4) ** 2 + (x_i[1] - 4) ** 2) ** 2)) / 1000 + 0.05)
        b = np.exp((-(((x_i[0] + 4) ** 2 + (x_i[1] + 4) ** 2) ** 2)) / 1000)
        c = np.exp(-(((x_i[0] + 4) ** 2 + (x_i[1] + 4) ** 2) ** 2))
        d = np.exp(-(((x_i[0] - 4) ** 2 + (x_i[1] - 4) ** 2) ** 2))
        return -(a + b + 0.15 * c + 0.15 * d)

    if f_index == 5:
        """Многоэкстремальная функция с оператором min.
        Область поиска экстремума [-6, 6]. Глобальный экстремум в точке: [-2, 4].
        Локальные экстремумы в порядке убывания значения функции: 
        [0, 0], [4, 4], [4, 0], [2, 0], [0, -2], [-4, 2], [2, -4], [2, 2], [-4, -2]"""
        # массив 10x120x120
        # истинный минимум в [-2 4]
        res = np.array([func5_1(x_i), func5_2(x_i), func5_3(x_i), func5_4(x_i), func5_5(x_i),
                        func5_6(x_i), func5_7(x_i), func5_8(x_i), func5_9(x_i), func5_10(x_i)])
        if len(x_i.shape) == 1:
            return np.min(res)
        else:
            Z = np.zeros((len(x_i[0]), len(x_i[1])))
            for i in range(len(res[0])):  # вдоль ширины
                for j in range(len(res[0][0])):  # вдоль глубины
                    Z[i][j] = np.min(res[:, i, j])
            return Z

    if f_index == 6:
        if dim == 2:
            return (1 - x_i[0]) ** 2 + 100 * (x_i[1] - x_i[0] ** 2) ** 2
        """Функция Розенброка
        Глобальный минимум = 0 в точке [1, 1]"""
        return np.sum(100 * (x_i[2:dim] - (x_i[1:dim - 1]) ** 2) ** 2 + (x_i[1:dim - 1] - 1) ** 2)

    if f_index == 7:
        """фунция Растригина
        Глобальный минимум = 0 в точке [0, 0]"""
        return np.sum(x_i**2 - 10 * np.cos(2 * np.pi * x_i) + 10)

    if f_index == 8:
        """Quartic function (Квартальная функция)
        Глобальный минимум = 0"""
        res = 0
        for i in range(len(x_i)):
            res = res + (i + 1) * x_i[i]**4
        return res + np.random.uniform(0, 1)

    if f_index == 9:
        """Griewank function
        Глобальный минимум = 0 в точке [0, 0]"""
        s = np.sum((x_i ** 2) / 4000)
        i = np.array([range(1, len(x_i) + 1)])
        p = np.prod(np.cos(x_i / (np.sqrt(i))))
        return s - p + 1

    if f_index == 10:
        """Функция Шекеля (Shekel’s Foxholes)
        Глобальный минимум 0,998 в точке [-32, -32]"""
        # a = np.array([[4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
        #               [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6]])
        # a = np.concatenate((a, a, a, a, a))
        # beta = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])

        A = [-32., -16., 0., 16., 32.]
        a1 = A * 5

        a2 = reduce(lambda x1, x2: x1 + x2, [[c] * 5 for c in A])

        if len(x_i.shape) == 1:
            x = x_i[0]
            y = x_i[1]
            r = 0.0
            for i in range(25):
                #           r += 1.0/ (1.0*i + pow(x-a1[i],6) + pow(y-a2[i],6) + 1e-15)
                z = 1.0 * i + pow(x - a1[i], 6) + pow(y - a2[i], 6)
                if z:
                    r += 1.0 / z
                # else:
                #     r += inf
            return 1.0 / (0.002 + r)

            # f = 0
            # for i in range(len(a[0])):
            #     sum_x = np.sum(x_i - a[:, i]) ** 2
            #     gamma = (sum_x + beta[i]) ** (-1)
            #     f = f + gamma
            # # return -(1 / 500 + b) ** (-1)
            # return -f
        else:
            Z = np.zeros((len(x_i[0]), len(x_i[1])))
            for i in range(len(x_i[0])):
                for j in range(len(x_i[1])):  # вдоль глубины

                    r = 0.0
                    for k in range(25):
                        #           r += 1.0/ (1.0*i + pow(x-a1[i],6) + pow(y-a2[i],6) + 1e-15)
                        z = 1.0 * i + pow(x_i[0, i, j] - a1[k], 6) + pow(x_i[1, i, j] - a2[k], 6)
                        if z:
                            r += 1.0 / z
                            # else:
                            #     r += inf

                    # f = 0
                    # for k in range(len(a[0])):
                    #     sum_x = np.sum(x_i[:, i, j] - a[:, k]) ** 2
                    #     gamma = (sum_x + beta[k]) ** (-1)
                    #     f = f + gamma
                    # Z[i][j] = -(1 / 500 + b) ** (-1)
                    Z[i][j] = r
            return Z

    if f_index == 11:
        """Многоэкстремальная функция. Глобальный минимум в точке [0, 0].
        Область поиска [-6, 6].
        Локальные минимумы по убыванию значения функции в точках: 
        [-2, 0], [0, -2], [0, 4], [2, 2], [4, 0], [4, 4], [-4, 4], [-4, -4], [3, -5]"""
        res = np.array([func11_1(x_i), func11_2(x_i), func11_3(x_i), func11_4(x_i), func11_5(x_i),
                        func11_6(x_i), func11_7(x_i), func11_8(x_i), func11_9(x_i), func11_10(x_i)])
        if len(x_i.shape) == 1:
            return np.min(res)
        else:
            Z = np.zeros((len(x_i[0]), len(x_i[1])))
            for i in range(len(res[0])):  # вдоль ширины
                for j in range(len(res[0][0])):  # вдоль глубины
                    Z[i][j] = np.min(res[:, i, j])
            return Z

    if f_index == 12:
        """Глобальный минимум в точке [9, 8.6]"""
        return x_i[0] * np.sin(4 * x_i[0]) + 1.1 * x_i[1] * np.sin(2 * x_i[1])

    if f_index == 13:
        """Глобальный минимум = 0 в точке [0, 0]"""
        f = np.sin(30 * ((x_i[0] + 0.5) ** 2 + x_i[1] ** 2) ** 0.1)
        return ((x_i[0] ** 2 + x_i[1] ** 2) ** 0.25) * f + np.abs(x_i[0]) + np.abs(x_i[1])

    if f_index == 14:
        """Глобальный минимум в точке [-5, -5]"""
        # res = np.array([func14_1(x_i), func14_2(x_i), func14_3(x_i), func14_4(x_i), func14_5(x_i),
        #                 func14_6(x_i), func14_7(x_i), func14_8(x_i), func14_9(x_i), func14_10(x_i)])
        res = np.array([func14_1(x_i), func14_2(x_i), func14_3(x_i), func14_4(x_i), func14_5(x_i)])
        if len(x_i.shape) == 1:
            return np.min(res)
        else:
            Z = np.zeros((len(x_i[0]), len(x_i[1])))
            for i in range(len(res[0])):  # вдоль ширины
                for j in range(len(res[0][0])):  # вдоль глубины
                    Z[i][j] = np.min(res[:, i, j])
            return Z

    if f_index == 15:
        """Многоэкстремальная функция. Глобальный минимум в точке [, ].
        Область поиска [-10, 10].
        Локальные минимумы по убыванию значения функции в точках: 
        [, ], [, ], [, ], [, ], [, ], [, ], [, ], [, ], [, ]"""
        res = np.array([func15_1(x_i), func15_2(x_i), func15_3(x_i), func15_4(x_i), func15_5(x_i),
                        func15_6(x_i), func15_7(x_i), func15_8(x_i), func15_9(x_i), func15_10(x_i)])
        if len(x_i.shape) == 1:
            return np.min(res)
        else:
            Z = np.zeros((len(x_i[0]), len(x_i[1])))
            for i in range(len(res[0])):  # вдоль ширины
                for j in range(len(res[0][0])):  # вдоль глубины
                    Z[i][j] = np.min(res[:, i, j])
            return Z

    if f_index == 16:
        """Многоэкстремальная функция. Глобальный минимум в точке [, ].
        Область поиска [-6, 6].
        Локальные минимумы по убыванию значения функции в точках: 
        [, ], [, ], [, ], [, ], [, ], [, ], [, ], [, ], [, ]"""
        res = np.array([func16_1(x_i), func16_2(x_i), func16_3(x_i), func16_4(x_i), func16_5(x_i),
                        func16_6(x_i), func16_7(x_i), func16_8(x_i), func16_9(x_i), func16_10(x_i)])
        if len(x_i.shape) == 1:
            return np.min(res)
        else:
            Z = np.zeros((len(x_i[0]), len(x_i[1])))
            for i in range(len(res[0])):  # вдоль ширины
                for j in range(len(res[0][0])):  # вдоль глубины
                    Z[i][j] = np.min(res[:, i, j])
            return Z

    if f_index == 17:
        a = (x_i[1] - (5.1 / (4 * np.math.pi**2)) * x_i[0]**2 + (5 / np.math.pi) * x_i[0] - 6)**2
        return a + 10 * (1 - 1 / (8 * np.math.pi) * np.cos(x_i[0])) + 10

    if f_index == 18:
        return (1 + (x_i[0] + x_i[1] + 1)**2 * (19 - 14*x_i[0]+3*(x_i[0]*x_i[0])-14*x_i[1]+6*x_i[0]*x_i[1]+3*x_i[1]*x_i[1]))*(30+(2*x_i[0]-3*x_i[1])**2*(18-32*x_i[0]+12*x_i[0]*x_i[0]+48*x_i[1]-36*x_i[0]*x_i[1]+27*x_i[0]*x_i[1]))

    if f_index == 19:
        """Многоэкстремальная функция. Глобальный минимум в точке [, ].
        Область поиска [-6, 6].
        Локальные минимумы по убыванию значения функции в точках: 
        [, ], [, ], [, ], [, ], [, ], [, ], [, ], [, ], [, ]"""
        res = np.array([func19_1(x_i), func19_2(x_i), func19_3(x_i), func19_4(x_i), func19_5(x_i),
                        func19_6(x_i), func19_7(x_i), func19_8(x_i), func19_9(x_i), func19_10(x_i), func19_11(x_i)])
        if len(x_i.shape) == 1:
            return np.min(res)
        else:
            Z = np.zeros((len(x_i[0]), len(x_i[1])))
            for i in range(len(res[0])):  # вдоль ширины
                for j in range(len(res[0][0])):  # вдоль глубины
                    Z[i][j] = np.min(res[:, i, j])
            return Z


def func5_1(x_i):
    return 6 * np.abs(x_i[0] + 2)**0.6 + 6 * np.abs(x_i[1] - 4)**1.6


def func5_2(x_i):
    return 6 * np.abs(x_i[0])**1.6 + 7 * np.abs(x_i[1])**2 + 3


def func5_3(x_i):
    return 6 * np.abs(x_i[0] - 4)**0.6 + 7 * np.abs(x_i[1] - 4)**0.6 + 5


def func5_4(x_i):
    return 5 * np.abs(x_i[0] - 4)**1.1 + 5 * np.abs(x_i[1])**1.8 + 6


def func5_5(x_i):
    return 5 * np.abs(x_i[0] + 2)**0.5 + 5 * np.abs(x_i[1])**0.5 + 7


def func5_6(x_i):
    return 5 * np.abs(x_i[0])**1.3 + 5 * np.abs(x_i[1] + 2)**1.3 + 8


def func5_7(x_i):
    return 4 * np.abs(x_i[0] + 4)**0.8 + 3 * np.abs(x_i[1] - 2)**1.2 + 9


def func5_8(x_i):
    return 2 * np.abs(x_i[0] - 2)**0.9 + 4 * np.abs(x_i[1] + 4)**0.3 + 10


def func5_9(x_i):
    return 6 * np.abs(x_i[0] - 2)**1.1 + 4 * np.abs(x_i[1] - 2)**1.7 + 11


def func5_10(x_i):
    return 3 * np.abs(x_i[0] + 4)**1.2 + 3 * np.abs(x_i[1] + 2)**0.5 + 12


def func11_1(x_i):
    return 6 * np.abs(x_i[0]) ** 2 + 7 * np.abs(x_i[1]) ** 2


def func11_2(x_i):
    return 5 * np.abs(x_i[0] + 2) ** 0.5 + 5 * np.abs(x_i[1]) ** 0.5 + 6


def func11_3(x_i):
    return 5 * np.abs(x_i[0]) ** 1.3 + 5 * np.abs(x_i[1] + 2) ** 1.3 + 5


def func11_4(x_i):
    return 4 * np.abs(x_i[0]) ** 0.8 + 3 * np.abs(x_i[1] - 4) ** 1.2 + 8


def func11_5(x_i):
    return 6 * np.abs(x_i[0] - 2) ** 1.1 + 4 * np.abs(x_i[1] - 2) ** 1.7 + 7


def func11_6(x_i):
    return 5 * np.abs(x_i[0] - 4) ** 1.1 + 5 * np.abs(x_i[1]) ** 1.8 + 9


def func11_7(x_i):
    return 6 * np.abs(x_i[0] - 4) ** 0.6 + 4 * np.abs(x_i[1] - 4) ** 0.6 + 4


def func11_8(x_i):
    return 6 * np.abs(x_i[0] + 4) ** 0.6 + 6 * np.abs(x_i[1] - 4) ** 1.6 + 3


def func11_9(x_i):
    return 3 * np.abs(x_i[0] + 4) ** 1.2 + 3 * np.abs(x_i[1] + 4) ** 0.5 + 7.5


def func11_10(x_i):
    return 2 * np.abs(x_i[0] - 3) ** 0.9 + 4 * np.abs(x_i[1] + 5) ** 0.3 + 8.5


def func14_1(x_i):
    # [-5, 2]
    return 4 * np.abs((x_i[0] + 5) ** 2) ** 0.1 + 2 * np.abs((x_i[1] - 2) ** 2) ** 0.2 + 7


def func14_2(x_i):
    # [1.5, 1.5] - локальный минимум
    return (3 * np.abs(x_i[0] - 1.5) + np.abs((x_i[1] - 1.5) ** 4) ** 0.3) * 1.5 + 1.5


def func14_3(x_i):
    # [-5, -5] - глобальный минимум
    return np.abs(x_i[0] + 5) ** 2.4 + np.abs(x_i[1] + 5) ** 2


def func14_4(x_i):
    # [3, -4]
    return 6 * np.abs(x_i[0] - 3) ** 1.1 + 4 * np.abs(x_i[1] + 4) ** 1.7 + 3.5


def func14_5(x_i):
    # [-3.7, 6]
    return 1.5 * np.abs(x_i[0] + x_i[1] * 0.6) ** 3.1 + 5 * np.abs(x_i[1] - 6) ** 3 + 5


# ###############################-----Функция 15-----#######################################
def func15_1(x_i):
    return 6 * np.abs(x_i[0] - 3) ** 0.5 + 6 * np.abs(x_i[1] + 5) ** 0.7


def func15_2(x_i):
    return 5 * np.abs(x_i[0] - 5) ** 2 + 5 * np.abs(x_i[1] - 5) ** 2 + 3


def func15_3(x_i):
    return 6 * np.abs(x_i[0] + 6) ** 0.9 + 6 * np.abs(x_i[1] - 3) ** 0.9 + 4


def func15_4(x_i):
    return 4 * np.abs(x_i[0] + 8) ** 1.6 + 7 * np.abs(x_i[1] + 8) ** 0.6 + 5


def func15_5(x_i):
    return 7 * np.abs(x_i[0]) ** 1.2 + 3 * np.abs(x_i[1] + 6) ** 1.3 + 6


def func15_6(x_i):
    return 5 * np.abs(x_i[0] + 4) ** 1.7 + 6 * np.abs(x_i[1] - 8) ** 2 + 7


def func15_7(x_i):
    return 2 * np.abs(x_i[0] + 5) ** 1.5 + 7 * np.abs(x_i[1] + 2) ** 1.1 + 8


def func15_8(x_i):
    return 6 * np.abs(x_i[0] - 9) ** 0.8 + 6 * np.abs(x_i[1]) ** 1.3 + 9


def func15_9(x_i):
    return 4 * np.abs(x_i[0] - 1) ** 1.9 + 4 * np.abs(x_i[1] - 1) ** 1.3 + 10


def func15_10(x_i):
    return 5 * np.abs(x_i[0] - 6) ** 0.6 + 7 * np.abs(x_i[1] + 7) ** 1.1 + 2.5


# ###############################-----Функция 16-----#######################################
def func16_1(x_i):
    return 7 * np.abs(x_i[0] - 2) ** 0.7 + 7 * np.abs(x_i[1] + 3) ** 0.9


def func16_2(x_i):
    return 4 * np.abs(x_i[0] + 4) ** 0.6 + 5 * np.abs(x_i[1] - 3) ** 1.2 + 2
    # return 4*np.abs(x_i[0] + 4) ** 0.3 + 5*np.abs(x_i[1] - 3) ** 0.3 + 2


def func16_3(x_i):
    return 6 * np.abs(x_i[0] - 4) ** 1.2 + 6 * np.abs(x_i[1] - 5) ** 0.3 + 2.5


def func16_4(x_i):
    return 5 * np.abs(x_i[0] + 2) ** 0.6 + 7 * np.abs(x_i[1] - 1) ** 1.3 + 4


def func16_5(x_i):
    return 3.5 * np.abs(x_i[0] + 3) ** 1.5 + 5 * np.abs(x_i[1] + 4) ** 2 + 5


def func16_6(x_i):
    return 7 * np.abs(x_i[0] + 5) ** 0.5 + 3 * np.abs(x_i[1] + 3) ** 0.9 + 6


def func16_7(x_i):
    return 6 * np.abs(x_i[0] - 4) ** 2 + 5 * np.abs(x_i[1] + 2) ** 0.6 + 6.5


def func16_8(x_i):
    return 3 * np.abs(x_i[0] - 2) ** 1.7 + 6.3 * np.abs(x_i[1] - 2) ** 1.1 + 7


def func16_9(x_i):
    return 4.5 * np.abs(x_i[0] - 3) ** 1.1 + 5 * np.abs(x_i[1] - 5) ** 0.8 + 8


def func16_10(x_i):
    return 2 * np.abs(x_i[0] + 1) ** 0.6 + 3 * np.abs(x_i[1] + 1) ** 1.1 + 9


# ###############################-----Функция 19-----#######################################
def func19_1(x_i):
    return 6 * np.abs(x_i[0] + 1) ** 0.6 + 6 * np.abs(x_i[1] - 3) ** 1.3


def func19_2(x_i):
    return 7 * np.abs(x_i[0] + 2) ** 0.8 + 3 * np.abs(x_i[1] + 2) ** 1.2 + 1.5


def func19_3(x_i):
    return 4 * np.abs(x_i[0] - 2) ** 1.3 + 5 * np.abs(x_i[1] + 3) ** 0.8 + 2


def func19_4(x_i):
    return 3 * np.abs(x_i[0] + 5) ** 0.9 + 6 * np.abs(x_i[1] + 5) ** 0.2 + 4


def func19_5(x_i):
    return 5 * np.abs(x_i[0] - 3) ** 1.4 + 7 * np.abs(x_i[1] - 1) ** 0.7 + 5


def func19_6(x_i):
    return 4 * np.abs(x_i[0]) ** 1.3 + 6 * np.abs(x_i[1] - 5) ** 0.8 + 6


def func19_7(x_i):
    return 7 * np.abs(x_i[0] - 5) ** 1.5 + 4 * np.abs(x_i[1] + 1) ** 0.5 + 8


def func19_8(x_i):
    return 4 * np.abs(x_i[0] + 4) ** 1.2 + 2 * np.abs(x_i[1] - 4) ** 0.6 + 9


def func19_9(x_i):
    return 3 * np.abs(-x_i[0] - 4) ** 0.4 + 3 * np.abs(-x_i[1] + 4) ** 0.3 + 2


def func19_10(x_i):
    return 3 * np.abs(-x_i[0] + 3) ** 0.7 + 2 * np.abs(-x_i[1] - 2) ** 1.1 + 3


def func19_11(x_i):
    return 4 * np.abs(-x_i[0] - 2) ** 1.6 + 5 * np.abs(-x_i[1] - 2) ** 1.1
