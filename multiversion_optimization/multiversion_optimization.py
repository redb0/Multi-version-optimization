import numpy as np

from GSA.GSA import GSA

from ICA.AlgorithmParams import AlgorithmParams
from ICA.ProblemParams import ProblemParams
from ICA.ica_alg import ICA

from SAC.sac_alg import SAC

from graph.graph import comparative_graph_convergence, grahp_isolines

# тестовые функции
from test_func.test_function_range import get_range
from test_func import test_function


def run_SAC(f_index, min_flag):
    max_iter = 600
    N = 50
    gamma = 1
    selectivity_factor = 150
    nuclear_func_index = 4
    q = 2
    epsilon = 5 * pow(10, -8)

    res = SAC(f_index, max_iter, N, min_flag, nuclear_func_index, selectivity_factor, gamma, q, epsilon)

    func_best, agent_best, best_chart, mean_chart, coord_x, stop_iteration, last_value_func, coord_x_test = res

    return func_best, agent_best, best_chart


def run_GSA(f_index, min_flag):
    N = 50
    max_iter = 600
    elitist_check = 1
    r_power = 1

    res = GSA(f_index, N, max_iter, elitist_check, min_flag, r_power)

    func_best, agent_best, best_chart, mean_chart, coord = res

    return func_best, agent_best, best_chart


def run_ICA(f_index, min_flag):
    num_of_countries = 200  # общее количество стран
    num_of_initial_imperialists = 10  # начальное количество империалистов
    revolution_rate = 0.3
    assimilation_coefficient = 2  # коэффициент ассимиляции "beta"
    assimilation_angle_coefficient = 0.5  # угловой коэффициент ассимиляции "gama"
    zeta = 0.02
    damp_ratio = 0.99
    stop_if_just_one_empire = False
    uniting_threshold = 0.02

    max_iter = 600
    epsilon = pow(10, -15)

    low, up, dim = get_range(f_index)
    ProblemParam = ProblemParams(dim, low, up, f_index, 0)

    AlgorithmParam = AlgorithmParams(num_of_countries, num_of_initial_imperialists, zeta,
                                     revolution_rate, damp_ratio, assimilation_coefficient,
                                     stop_if_just_one_empire, uniting_threshold)

    res = ICA(ProblemParam, AlgorithmParam, max_iter, min_flag, epsilon)

    func_best, agent_best, best_chart, mean_chart, coord_x, stop_iteration = res

    return func_best, agent_best, best_chart


def optimization(f_index, min_flag):
    print("Запущен алгоритм селективного усреднения координат...")
    func_best_SAC, agent_best_SAC, best_chart_SAC = run_SAC(f_index, min_flag)
    print("Запущен алгоритм гравитационного поиска...")
    func_best_GSA, agent_best_GSA, best_chart_GSA = run_GSA(f_index, min_flag)
    print("Запущен империалистический конкурентный алгоритм...")
    func_best_ICA, agent_best_ICA, best_chart_ICA = run_ICA(f_index, min_flag)

    func_best = np.array([func_best_SAC, func_best_GSA, func_best_ICA])
    agent_best = np.array([agent_best_SAC, agent_best_GSA, agent_best_ICA])

    comparative_graph_convergence(f_index, SAC=best_chart_SAC, GSA=best_chart_GSA, ICA=best_chart_ICA)

    grahp_isolines(f_index)

    # print(str(func_best))
    # print(str(agent_best))

    # func_best, agent_best, best_chart, mean_chart, coord_x, stop_iteration, last_value_func, coord_x_test = result_SAC
    # func_best, agent_best, best_chart, mean_chart, coord = result_GSA
    # func_best, agent_best, best_chart, mean_chart, coord_x, stop_iteration = result_ICA

    low, up, dim = get_range(f_index)

    while len(func_best) > 1:

        min_fit = np.min(func_best)
        arg_min = np.argmin(func_best)

        if np.all(agent_best[arg_min] > low) and np.all(agent_best[arg_min] < up):
            min_test = test_function.test_function(agent_best[arg_min], f_index, dim)
            if min_test <= min_fit:
                print("Лучший результат: " + str(min_test))
                print("Лучшая точка: " + str(agent_best[arg_min]))
                if arg_min == 0:
                    print("Алгоритм селективного усреднения")
                elif arg_min == 1:
                    print("Алгоритм гравитационного поиска")
                elif arg_min == 2:
                    print("Империалистический конкурентный алгоритм")
                break
        else:
            func_best = np.delete(func_best, [arg_min])
            agent_best = np.delete(agent_best, [arg_min])

    print("Работа завершена.")


def main():
    # 8, 9 - 30 мерная
    f_index = 4
    min_flag = 1

    optimization(f_index, min_flag)


if __name__ == "__main__":
    main()
