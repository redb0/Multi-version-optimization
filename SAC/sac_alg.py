import numpy as np

import graph

from SAC.nuclear_function import get_function

# тестовые функции
from test_func import test_function_range
from test_func import test_function


# расположение центральной точки
# def initialization_operating_point(dimension, up, down):
#     pass


# расположение тестовых точек
def initialization_test_points(N, dimension, delta_X, X):
    # delta_X - одномерный массив из одной строки и dimension столбцов,
    # в строке находится ширина области поиска

    test_points = np.random.uniform(0, 1, (N, dimension))

    for i in range(N):
        high = X + delta_X  # верхняя граница поиска
        low = X - delta_X  # нижняя граница поиска
        test_points[i, :] = test_points[i, :] * (high - low)
        test_points[i, :] = test_points[i, :] + low

    return test_points


def initialization_X_0(low, up, dim):

    X_0 = np.zeros((dim, ))

    if (type(up) == int) or (type(up) == float):
        down = low
        high = up
    # else:
    #     down = low
    #     high = up

    for i in range(dim):
        X_0[i] = (high - down) / 2
        X_0[i] = X_0[i] + down

    return X_0


# def initialization_delta(low, up, X_0):
#     dim = len(X_0)
#
#     if (type(up) == int) or (type(up) == float):
#         down = low
#         high = up
#
#     for i in range(dim):


# движение центральной точки
def move(delta_X, X, test_points, nuclear_func, q):
    N = len(test_points)
    dim = len(X)

    u = np.zeros((N, dim))

    delta = np.array([0 for i in range(dim)])
    f_coord = np.array([0 for i in range(dim)])

    for i in range(N):
        u[i] = (test_points[i] - X) / (delta_X + np.finfo(float).eps)

        delta = delta + (nuclear_func[i] * np.abs(u[i]) ** q)

        f_coord = f_coord + nuclear_func[i] * u[i]

    delta_X = delta_X * delta ** (1 / q)
    X = X + f_coord * delta_X

    # u = np.zeros((dim, ))
    #
    # for i in range(dim):
    #     random_mas = np.random.uniform(-1, 1, (len(nuclear_func),))
    #     u[i] = np.sum(random_mas * nuclear_func)
    #
    # # новая координата центральной точки
    # X = X + delta_X * u

    #
    # delta_X = gamma * delta_X * (np.sum((np.abs(np.random.uniform(-1, 1))**q) * nuclear_func))**(1 / q)
    # delta_X = gamma * delta_X * (np.sum((np.abs(np.random.uniform(-1, 1, (1, len(nuclear_func)))) ** q) * nuclear_func)) ** (1 / q)

    return X, delta_X


def size_area_search_calculation(delta_X, gamma, q, nuclear_func, N):
    dim = len(delta_X)
    u = np.random.uniform(-1, 1, (N, dim))

    sum_u = np.zeros((dim, ))

    for i in range(dim):
        sum_u[i] = np.sum((np.abs(u[:, i]) ** q) * nuclear_func)

    delta_X = gamma * delta_X * sum_u ** (1 / q)

    return delta_X


def find_g(test_points, fitness_test_points, min_flag):
    fit_max_tp = np.max(fitness_test_points)
    fit_min_tp = np.min(fitness_test_points)

    # i = len(fit)
    N = len(fitness_test_points)
    dimention = len(test_points[0])

    if fit_min_tp == fit_max_tp:
        g = np.ones((N, 1))
    else:
        if min_flag == 1:  # минимизация
            best = fit_min_tp
            worst = fit_max_tp
            g = (fitness_test_points - best) / (worst - best)
        else:  # максимизация
            best = fit_max_tp
            worst = fit_min_tp
            g = (best - fitness_test_points) / (best - worst)

    return g


def nuclear_function_K(g, nuclear_func_index, selectivity_factor):

    nuclear_func = get_function(g, nuclear_func_index, selectivity_factor)

    nuclear_func = nuclear_func / np.sum(nuclear_func)

    return nuclear_func


def evaluate_function(X, test_points, f_index):
    N = len(test_points)
    dim = len(test_points[0])

    fitness_test_points = np.zeros(N)

    # расчет значения функции в центральной точке
    fitness_operating_point = test_function.test_function(X, f_index, dim)

    for i in range(N):
        x_i = test_points[i, :]
        fitness_test_points[i] = test_function.test_function(x_i, f_index, dim)

    return fitness_test_points, fitness_operating_point


def SAC(f_index, max_iter, N, min_flag, nuclear_func_index, selectivity_factor, gamma, q, epsilon):
    low, up, dimension = test_function_range.get_range(f_index)

    stop_iteration = max_iter
    best_chart = []
    mean_chart = []

    coord_x = 0
    coord_x_test = 0

    if dimension == 2:
        # массив для сохранения прогресса координат решений
        coord_x = np.empty((max_iter, 1, dimension))
        coord_x_test = np.empty((max_iter, N, dimension))

    X = initialization_X_0(low, up, dimension)

    delta_X_0 = np.array([(up - low) / 2])

    # delta_X = np.zeros((dimension, ))

    delta_X = delta_X_0

    for i in range(max_iter):
        iteration = i + 1

        if iteration == 1:
            test_points = initialization_test_points(N, dimension, delta_X_0, X)
            fitness_test_points, fitness_operating_point = evaluate_function(X, test_points, f_index)
        else:
            test_points = initialization_test_points(N, dimension, delta_X, X)
            fitness_test_points, fitness_operating_point = evaluate_function(X, test_points, f_index)

        if dimension == 2:
            coord_x[i] = X.copy()
            coord_x_test[i] = test_points.copy()

        if min_flag == 1:
            # лучшее решение
            best = np.min(fitness_test_points)
            # лучшее положение (точка, агент)
            best_x = np.argmin(fitness_test_points)
        else:
            best = np.max(fitness_test_points)
            best_x = np.argmax(fitness_test_points)

        if iteration == 1:
            # лучшее значение функции
            func_best = best
            # лучший агент
            agent_best = test_points[best_x, :]

        if min_flag == 1:
            # минимизация
            if best < func_best:
                func_best = best
                agent_best = test_points[best_x, :]
        else:
            # максимизация
            if best > func_best:
                func_best = best
                agent_best = test_points[best_x, :]

        # сохранение лучших и средних решений на итерации
        # best_chart.append(func_best)
        best_chart.append(fitness_operating_point)
        mean_chart.append(np.mean(fitness_test_points))

        g = find_g(test_points, fitness_test_points, min_flag)

        nuclear_func = nuclear_function_K(g, nuclear_func_index, selectivity_factor)

        X, delta_X = move(delta_X, X, test_points, nuclear_func, q)

        # delta_X = size_area_search_calculation(delta_X, gamma, q, nuclear_func, N)

        last_value_func = fitness_operating_point

        # if iteration > 2:
        #     break_point = break_check(best_chart, epsilon, delta_X)
        #     if break_point:
        #         stop_iteration = iteration
        #         break

    return func_best, agent_best, best_chart, mean_chart, coord_x, stop_iteration, last_value_func, coord_x_test


# Функция для проверки условий останова
def break_check(best_chart, epsilon, delta_X):
    e = pow(10, -10)
    k = 0
    for i in range(len(delta_X)):
        if delta_X[i] < e:
            k = k + 1
    if k * 1.0 >= len(delta_X) * 0.9:
        return True

    if np.abs(best_chart[len(best_chart) - 1] - best_chart[len(best_chart) - 2]) <= epsilon:
        return True
    else:
        return False


def main():

    f_index = 5
    N = 50
    max_iter = 100
    min_flag = 1
    nuclear_func_index = 3
    rate_change_graph = 700
    epsilon = pow(10, -10)

    selectivity_factor = 200
    gamma = 1
    q = 2

    func_best, agent_best, best_chart, mean_chart, coord_x, stop_iteration, last_value_func, coord_x_test = SAC(f_index, max_iter, N, min_flag, nuclear_func_index, selectivity_factor, gamma, q, epsilon)

    print("Лучшее значение тестовой функции из всех значений тестовых точек: " + str(func_best))
    print("Лучшее решение: " + str(agent_best))

    print("Последнее значений функции (в момент останова): " + str(last_value_func))
    print("Координаты: " + str(coord_x[len(coord_x) - 1]))

    print(str(stop_iteration))

    graph.graph_motion_points_3d(f_index, rate_change_graph, coord_x, stop_iteration)

    graph.graph_best_chart(best_chart)

    graph.print_graph(f_index, rate_change_graph, coord_x, stop_iteration)

    graph.print_graph(f_index, rate_change_graph, coord_x_test, stop_iteration)


if __name__ == "__main__":
    main()
