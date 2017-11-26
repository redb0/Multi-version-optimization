import math
import numpy as np

import graph.graph

# тестовые функции
from test_func import test_function_range
from test_func import test_function


# Функция вычисления Гравитационной постоянной.
# Должна изменяться по монотонно убывающей функции.
# @param iteration - номер итерации
# @param max_iter  - максимальное число итераций
# @return G        - гравитационная постоянная, число
def G_constant(iteration, max_iter):
    # здесь можно задать свою монотонно убывающую функцию
    # например
    # d = 0.5
    # G = G0 * math.pow(max_iter / iteration, d)
    G0 = 100
    alfa = 20

    G = G0 * math.exp(-alfa * iteration / max_iter)
    return G


# Инициализация матрицы X начальных агентов.
# Матрица имеет размерность N (строк) на dimension (столбцов, измерения).
# Вид матрицы X:
# [ [ x11; x12; ...; x1d ]
#   [ x21; x22; ...; x2d ]
#   ...
#   [ xN1; xN2; ...; xNd ] ]
# @param dimension - количество измерений (переменных)
# @param N         - количество агентов (точек для поиска)
# @param up        - верхняя граница поиска, представляет либо одиночное значение, либо вектор
# @param down      - нижняя граница поиска, представлеет собой одиночное значение, либо вектор
# @return X        - матрица инициализированных точек (агентов)
def initialization(dimension, N, up, down):
    X = np.random.uniform(0, 1, (N, dimension))
    # если ограничения для всех измерений одинаковы и представляют одиночное значение
    # if len(up) == 1:
    if (type(up) == int) or (type(up) == float):
        X = X * (up - down)
        X = X + down
    # если ограничения для измерений разные и записаны в вектор
    else:
        if len(up) > 0:
            for i in range(dimension):
                high = up[i]   # верхняя граница поиска
                low = down[i]  # нижняя граница поиска
                # срезы по столбцам X[:, i] - i-й столбец в матрице X
                X[:, i] = X[:, i] * (high - low)
                X[:, i] = X[:, i] + low

    return X


# Функция проверки выхода агентов за границы поиска.
# Заного инициализирует агентов, которые вышли за границы поиска.
# Можно не инициализировать их заного, а возвращать к границам.
# @param X   - матрица точек (агентов)
# @param up  - верхняя граница поиска
# @param low - нижняя граница поиска
# @return X  - матрица точек (агентов), где заного инициализированы те агенты, которые вышли за границу поиска
def space_bound(X, up, low):
    N = len(X)
    dim = len(X[0])
    for i in range(N):
        # проверка выхода i-го агента за нижнюю границу поиска.
        # high_border, low_border - булевый массив, где False - те кто не вышел за границу, True - те кто вышел
        high_border = X[i, :] > up
        low_border = X[i, :] < low
        # ~ - операция конвертации значения в противоположное
        X[i, :] = (X[i, :] * (~(low_border + high_border))) + ((np.random.uniform(0, 1, (1, dim)) * (up - low) + low) * (low_border + high_border))

    return X


# Функция качества.
# В качестве кретерия оптимальности выступает само значение оптимизируемой функции.
# @param X       - матрица точек (агентов)
# @param f_index - номер (индекс) тестовой функции
# @return fit    - массив значений тестовой функции в каждой из N точек
def evaluate_function(X, f_index):
    N = len(X)
    dim = len(X[0])

    fit = np.zeros(N)

    for i in range(N):
        x_i = X[i, :]
        fit[i] = test_function.test_function(x_i, f_index, dim)

    return fit


# Функция для расчета масс.
#
# @param fit      - массив значений качественного критерия (оптимизируемой функции)
# @param min_flag - флаг, 1 - минимизация, 0 - максимизация
# @return mass    - массив масс для каждой иточки (агента)
def mass_calculation(fit, min_flag):
    fit_max = np.max(fit)
    fit_min = np.min(fit)
    fit_mean = np.mean(fit)

    # i = len(fit)
    N = len(fit)

    if fit_max == fit_min:
        mass = np.ones((N, 1))
    else:
        if min_flag == 1:  # минимизация
            best = fit_min
            worst = fit_max
        else:  # максимизация
            best = fit_max
            worst = fit_min

        mass = (fit - worst) / (best - worst)

    mass = mass / np.sum(mass)

    return mass


# Функция для расчета ускорения.
#
#
# @param X             - массив координат по оси x каждой из N точек (агентов)
# @param mass          - массив масс каждой из N точек (агентов)
# @param G             - значение гравитационной постоянной
# @param r_norm        - нормирование евклидова расстояния
# @param r_power       - степень расстояния в формуле закона тяготения
# @param elitist_check - флаг, 1- расчет происходит по формуле 21, 0 по формуле 9
# @param iteration     - номер текущей итерации
# @param max_iter      - общее количество итерациий алгоритма
# @return a            - значение ускорения
def acceleration_calc(X, mass, G, r_norm, r_power, elitist_check, iteration, max_iter):
    N = len(X)
    dim = len(X[0])

    final_per = 2

    if elitist_check == 1:
        kbest = final_per + (1 - iteration / max_iter) * (100 - final_per)
        kbest = round(N * kbest / 100)
    else:
        kbest = N

    # сортирует массив масс и возвращает отсортированный массив в виде его индексов
    ds = np.argsort(mass)[::-1]

    E = np.zeros((N, dim))

    for i in range(N):
        for ii in range(kbest):
            j = ds[ii]
            if j != i:
                radius = np.linalg.norm(X[i, :] - X[j, :], r_norm)
                for k in range(dim):
                    # np.finfo(float).eps - машинная погрешность. для float64 = 2^-16
                    E[i, k] = E[i, k] + np.random.uniform(0, 1) * mass[j] * ((X[j, k] - X[i, k]) / (math.pow(radius, r_power) + np.finfo(float).eps))

    a = E * G
    return a


# Функция движения.
# Рассчитывает скорость и новые координаты агентов
#
# @param X         - массив координат агентов по оси x
# @param a         - массив ускорений агентов
# @param velocity  - скорость агентов
# @return X        - массив новых положений агентов
# @return velocity - массив скорости каждого агента
def move(X, a, velocity):
    N = len(X)
    dim = len(X[0])

    velocity = np.random.uniform(0, 1, (N, dim)) * velocity + a
    X = X + velocity

    return X, velocity


# Функция реализующая алгоритм гравитационного поиска.
#
#
# @param f_index       - индекс тестовой функции в файле test_function
# @param N             - количество агентов
# @param max_iter      - общее количество итераций алгоритма
# @param elitist_check - флаг, 1 - расчет идет по 21 формуле, 0 - по 9 формуле
# @param min_flag      - флаг, обозначающий минимизацию либо максимизацию (1 - минимизация, 0 - максимизация)
# @param r_power       - степень расстояния в формуле закона тяготения
# @return func_best    - лучший результат (значение минимизируемой или максимизируемой функции)
# @return agent_best   - лучшее решение (точка, агент). Местоположение func_best на оси x
# @return best_chart   - массив лучших решений на каждой итерации
# @return mean_chart   - массив средних решений на каждой итерации
def GSA(f_index, N, max_iter, elitist_check, min_flag, r_power):
    r_norm = 2
    low, up, dimension = test_function_range.get_range(f_index)

    # if dimension == 2:
        # массив для сохранения прогресса координат решений
    coord = np.empty((max_iter, N, dimension))

    X = initialization(dimension, N, up, low)

    velocity = np.zeros((N, dimension))

    best_chart = []
    mean_chart = []

    for i in range(max_iter):
        iteration = i + 1

        # print("Началась " + str(iteration) + " итерация алгоритма.")

        # проверка выхода за границы поиска
        X = space_bound(X, up, low)
        # расчет значений функции качества (минимизируемой или максимизируемой функции)
        fit = evaluate_function(X, f_index)

        if dimension == 2:
            coord[i] = X.copy()

        if min_flag == 1:
            # лучшее решение
            best = np.min(fit)
            # лучшее положение (точка, агент)
            best_x = np.argmin(fit)
        else:
            best = np.max(fit)
            best_x = np.argmax(fit)

        if iteration == 1:
            # лучшее значение функции
            func_best = best
            # лучший агент
            agent_best = X[best_x, :]

        if min_flag == 1:
            # минимизация
            if best < func_best:
                func_best = best
                agent_best = X[best_x, :]
        else:
            # максимизация
            if best > func_best:
                func_best = best
                agent_best = X[best_x, :]

        # сохранение лучших и средних решений на итерации
        best_chart.append(func_best)
        mean_chart.append(np.mean(fit))

        # расчет масс, гравитационной постоянной, ускорения
        mass = mass_calculation(fit, min_flag)
        G = G_constant(iteration, max_iter)
        a = acceleration_calc(X, mass, G, r_norm, r_power, elitist_check, iteration, max_iter)
        # расчет скорости и нового положения
        X, velocity = move(X, a, velocity)

    return func_best, agent_best, best_chart, mean_chart, coord


def main():
    N = 50
    max_iter = 100
    elitist_check = 1
    r_power = 1
    min_flag = 1  # 1 - минимизация, 0 - максимизация
    # скорость изменения графика, установите значение в милисекундах
    rate_change_graph = 500

    # индекс функции в файле test_function
    # 18 не работает построение последнего графика.
    f_index = 19

    print("Подождите идут вычисления...")

    func_best, agent_best, best_chart, mean_chart, coord = GSA(f_index, N, max_iter, elitist_check, min_flag, r_power)

    print("-------------------------------")
    print("Лучший результат: " + str(func_best))
    print("Лучшее решение (точка, агент): " + str(agent_best))
    print("Лучшие решения на каждой итерации: " + str(best_chart))
    print("Среднее решений на каждой итерации: " + str(mean_chart))
    print("-------------------------------")

    graph.graph.graph_motion_points_3d(f_index, rate_change_graph, coord, max_iter)

    # graph.graph.graph_best_chart(best_chart)

    graph.graph.print_graph(f_index, rate_change_graph, coord, max_iter)


if __name__ == "__main__":
    main()
