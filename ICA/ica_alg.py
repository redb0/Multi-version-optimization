import numpy as np

from ICA import AlgorithmParams, Empires, ProblemParams

from graph import graph

# тестовые функции
from test_func import test_function_range
from test_func import test_function

"""
-------------------------------------------------------
Реализация Империалистического конкурентного алгоритма.
        (Imperialist Competitive Algorithm)
Алгоритм описан в статье:
Imperialist Competitive Algorithm: An Algorithm for Optimization Inspired by Imperialistic Competition.

Шаги алгоритма:
1. Инициализация империй:
    1.1. Генерация стран;
    1.2. Расчет стоимости (приспособленности) стран;
    1.3. Распределение стран (колоний) среди империалистов
    в зависимости от их силы, чем больше сила тем больше колоний получает империалист.
2. Перемещение колоний в строну их соответствующих империалистов:
    2.0 Оптимизация положения империалиста:
        2.0.1 Выбрать случайным образом малый размер области вокруг империалиста;
        2.0.2 Случайно выбрать точку внути этой области;
        2.0.3 Если значений функции качества в этой точке меньше текущего значения 
        функции качества империалиста, то переместить его в эту точку;
    2.1 Ассимиляция;
    2.2 Революция.
3. Если в империи есть колония с меньшим значением функции качетва 
то поменять ее с текущим империалистом.
4. Вычислить общую стоимость каждой империи.
5. Империалистическая конкуренция:
    5.1 Выбрать самую слабую империю
    5.2 Передать случайную колонию более сильной империи
6. Если есть империя без колоний, то устранить эту империю.
7. Если условие останова не выполенено вурнуться к шагу 2.
"""


# Функция преобразования границ поиска к единому виду (массив-строка),
# расчет размера области поиска.
# @param min_limit_search   - нижняя граница поиска (число либо одномерный массив-строка)
# @param max_limit_search   - верхняя граница поиска (число либо одномерный массив-строка)
# @param dimension          - размерность задачи оптимизации (пространства)
# @return min_limit         - нижняя граница поиска (массив-строка)
# @return max_limit         - верхняя граница поиска (массив-строка)
# @return search_space_size - размер области поиска (массив-строка)
def modifying_size_search_space(min_limit_search, max_limit_search, dimension):
    """
    Функция преобразования границ области поиска к единому виду.
    Кроме этого поисходит расчет размера области поиска.
    Если границы поиска экстремума (min_limit_search и max_limit_search) представлены в виде чисел,
    то происходит преобразование в одномерный массив-строку (numpy) длины dimension с одинаковыми элементами.
    Если границы изначально заданы в виде массива, то никаких преобразований не происходит.
    Параметры:
    :param min_limit_search: Нижняя (минимальная) граница области поиска экстремума. 
            Может быть представлена в виде числа типа float или int, 
            либо в виде одномерного массива-строки длиной dimension.
    :param max_limit_search: Верхняя (максимальная) граница области поиска экстремума.
            Может быть представлена в виде числа типа float или int, 
            либо в виде одномерного массива-строки длиной dimension.
    :param dimension: Размерность (количество переменных) целевой функции. 
            Может быть представлена в виде числа типа int.
            
    :return min_limit: Нижняя граница области поиска экстремума. 
            Представлена одномерным массивом-строкой (numpy) длины dimension.
    :return max_limit: Верхняя граница области поиска экстремума
            Представлена одномерным массивом-строкой (numpy) длины dimension.
    :return search_space_size: Размер области поиска. Длина отрезка для каждого измерения.
            Представлена одномерным массивом-строкой (numpy) длины dimension.
    """
    if (type(min_limit_search) == int) or (type(min_limit_search) == float):
        min_limit = np.array([min_limit_search for i in range(dimension)])
        max_limit = np.array([max_limit_search for i in range(dimension)])
    else:
        min_limit = min_limit_search
        max_limit = max_limit_search

    search_space_size = max_limit - min_limit

    return min_limit, max_limit, search_space_size


# Функция генерации координат новых стран.
# @param num_of_countries - исходное количество стран
# @param min_limit_search - массив-строка нижняя граница области поиска
# @param max_limit_search - массив-строка, верхняя граница области посика
# @return new_country     - массив координат новых стран
def generate_new_country(num_of_countries, min_limit_search, max_limit_search):
    """
    Функция для генерации координат новых стран (точек).
    
    :param num_of_countries: Количество стран. Целое число.
    :type num_of_countries: int.
    
    :param min_limit_search: Нижняя граница области поиска экстремума.
    :type min_limit_search: numpy.array
    
    :param max_limit_search: Верхняя граница области поиска экстремума.
    :type max_limit_search: numpy.array
    
    :return new_country: Двумерный массив с координатами стран (точек). 
            Строки - страны, столбцы - координаты для каждого измерения.
    """
    min_limit_matrix = np.array([min_limit_search for i in range(int(num_of_countries))])
    max_limit_matrix = np.array([max_limit_search for i in range(int(num_of_countries))])
    random_matrix = np.random.uniform(0, 1, min_limit_matrix.shape)
    new_country = random_matrix * (max_limit_matrix - min_limit_matrix) + min_limit_matrix
    return new_country


# Функция создания начальных империй (один империалист и колонии)
# @param initial_сountries       -
# @param initial_fitness_country -
# @param ProblemParam            -
# @return empires                -
def create_initial_empires(initial_сountries, initial_fitness_country, ProblemParam, AlgorithmParam):
    """
    Функция инициализации империй (точек с лучшим значенийм функции качества (стоимостью)).
    
    Параметры:
    :param initial_сountries: Набор координат стран, составленный в порядке убывания их значения функции качества. 
            Двумерный массив. 
    :type initial_сountries: numpy.array
    
    :param initial_fitness_country: Набор значений функции качетва (стоимостей) стран, 
            отсортированный в порядке убывания. Одномерный массив-строка.
    :type initial_fitness_country: numpy.array
    
    :param ProblemParam: Объект хранящий параметры задачи оптимизации. Объект класса ProblemParams.
            Имеет поля:
            dimention:           Размерность (количество переменных) целевой функции типа int.
            min_limit_search:    Нижняя граница области поиска экстремума. Одномерный массив-строка типа numpy.array.
            max_limit_search:    Верхняя граница области поиска экстремума. Одномерный массив-строка типа numpy.array.
            test_function_index: Номер тестовой функции из файла test_function.
            search_space_size:   Размер области поиска экстремума. Одномерный массив-строка типа numpy.array.
    :type ProblemParam: ProblemParams
    
    :param AlgorithmParam: Объект хранящий параметры алгоритма. Объект класса AlgorithmParams.
            Имеет поля:
            num_of_initial_imperialists: Изначальное количество империалистов, целое число.
            num_of_countries:            Количество стран, целое число.
            zeta:                        Коэффициент для расчета общей ценности империи, 
                                         характеризует влияние стоимостей колоний на общую стоимость империи.
            revolution_rate:             ?????
            damp_ratio:                  Процент колиний подвернувшихся революции.
            assimilation_coefficient:    Коэффициент ассимиляции, 
                                         характеризует процент максимально возможного шага колонии за одну итерацию.
            stop_if_just_one_empire:     True - алгоритм можно завершить если останеться только один империалист,
                                         False - иначе.
            uniting_threshold:           Величина расстояния между двумя империалистами 
                                         при которой произойдет слияние их империй.
    :type AlgorithmParam: AlgorithmParams
    
    :return empires: Список хрянящий объекты класса Empires.
    """
    num_all_colonies = AlgorithmParam.num_of_countries - AlgorithmParam.num_of_initial_imperialists
    num_init_imper = AlgorithmParam.num_of_initial_imperialists

    all_imperialists_position = initial_сountries[:num_init_imper]
    all_imperialists_fitness = initial_fitness_country[:num_init_imper]

    all_colonies_position = initial_сountries[num_init_imper:]
    all_colonies_fitness = initial_fitness_country[num_init_imper:]

    if np.max(all_imperialists_fitness) > 0:
        all_imperialists_power = 1.3 * np.max(all_imperialists_fitness) - all_imperialists_fitness
    else:
        all_imperialists_power = 0.7 * np.max(all_imperialists_fitness) - all_imperialists_fitness

    power_norm = all_imperialists_power / np.sum(all_imperialists_power)
    all_imperialists_num_colonies = np.round(power_norm * num_all_colonies)
    # последнее значение это число всех стран за минусом уже распределенных
    all_imperialists_num_colonies[-1] = num_all_colonies - np.sum(all_imperialists_num_colonies[:-1])

    random_index = np.random.permutation(num_all_colonies)
    empires = []

    for i in range(AlgorithmParam.num_of_initial_imperialists):
        imp_pos = all_imperialists_position[i]
        imp_fit = all_imperialists_fitness[i]
        rand = random_index[:int(all_imperialists_num_colonies[i])]
        colonies_pos = all_colonies_position[rand]
        colonies_fit = all_colonies_fitness[rand]
        total_fit = imp_fit + AlgorithmParam.zeta * np.mean(colonies_fit)
        Empire = Empires.Empires(imp_pos, imp_fit, colonies_pos, colonies_fit, total_fit)
        empires.append(Empire)

    for i in range(len(empires)):
        if len(empires[i].colonies_position) == 0:
            empires[i].colonies_position = generate_new_country(1, ProblemParam.min_limit_search, ProblemParam.max_limit_search)
            empires[i].colonies_fitness = [
                test_function.test_function(empires[i].colonies_position[0], ProblemParam.test_function_index, ProblemParam.dimention)]

    return empires


def assimilate_colonies(empire, ProblemParam, AlgorithmParam):
    num_colonies = len(empire.colonies_position)
    vector = np.array([empire.colonies_position[0] for i in range(num_colonies)])
    vector = vector - empire.colonies_position

    delta = AlgorithmParam.assimilation_coefficient * np.random.uniform(0, 1, vector.shape) * vector
    empire.colonies_position = empire.colonies_position + 2 * delta

    min_matrix = np.array([ProblemParam.min_limit_search for i in range(num_colonies)])
    max_matrix = np.array([ProblemParam.max_limit_search for i in range(num_colonies)])

    for i in range(num_colonies):
        for j in range(len(empire.colonies_position[i])):
            empire.colonies_position[i][j] = max(empire.colonies_position[i][j], min_matrix[i][j])
            empire.colonies_position[i][j] = min(empire.colonies_position[i][j], max_matrix[i][j])

    return empire


def revolve_colonies(empire, ProblemParam, AlgorithmParam):
    if (type(empire.colonies_fitness) == 'numpy.float64') or (type(empire.colonies_fitness) == int):
        lenght = 1
    else:
        lenght = len(empire.colonies_fitness)
    num_revolving_colonies = int(np.round(AlgorithmParam.revolution_rate * lenght))
    min_limit = ProblemParam.min_limit_search
    max_limit = ProblemParam.max_limit_search
    revolved_position = generate_new_country(num_revolving_colonies, min_limit, max_limit)
    rand = np.random.permutation(lenght)
    if lenght != 1:
        rand = rand[:num_revolving_colonies]
        for i in range(len(rand)):
            empire.colonies_position[rand[i]] = revolved_position[i]
    else:
        empire.colonies_position = revolved_position

    return empire


def prosses_empire(empire):
    colonies_fit = empire.colonies_fitness
    if len(colonies_fit) == 0:
        return empire

    min_colonies_fit = np.min(colonies_fit)
    index_min_fit = np.argmin(colonies_fit)

    if min_colonies_fit < empire.imperialist_fitness:
        old_imperialist_position = empire.imperialist_position
        old_imperialist_fit = empire.imperialist_fitness

        empire.imperialist_position = empire.colonies_position[index_min_fit].copy()
        empire.imperialist_fitness = empire.colonies_fitness[index_min_fit]

        empire.colonies_position[index_min_fit] = old_imperialist_position
        empire.colonies_fitness[index_min_fit] = old_imperialist_fit

        # empire.imperialist_position, empire.colonies_position[index_min_fit] = empire.colonies_position[index_min_fit], empire.imperialist_position
        # empire.imperialist_fitness, empire.colonies_fitness[index_min_fit] = empire.colonies_fitness[index_min_fit], empire.imperialist_fitness

    return empire


def unite_simular_empires(empires, ProblemParam, AlgorithmParam):
    threshold_distance = AlgorithmParam.uniting_threshold * np.linalg.norm(ProblemParam.search_space_size)
    num_empires = len(empires)

    for i in range(num_empires - 1):
        for j in range(i + 1, num_empires):
            distance_vector = empires[i].imperialist_position - empires[j].imperialist_position
            distance = np.linalg.norm(distance_vector)

            if distance <= threshold_distance:
                if empires[i].imperialist_fitness < empires[j].imperialist_fitness:
                    better_empire_ind = i
                    worse_empire_ind = j
                else:
                    better_empire_ind = j
                    worse_empire_ind = i

                toVstack = [empires[better_empire_ind].colonies_position,
                            empires[worse_empire_ind].imperialist_position,
                            empires[worse_empire_ind].colonies_position]
                empires[better_empire_ind].colonies_position = np.vstack(tuple(filter(lambda x: x.size != 0, toVstack)))
                del toVstack

                toVstack = [empires[better_empire_ind].colonies_fitness,
                            empires[worse_empire_ind].imperialist_fitness,
                            empires[worse_empire_ind].colonies_fitness]

                empires[better_empire_ind].colonies_fitness = np.hstack(tuple(filter(lambda x: x.size != 0, toVstack)))
                del toVstack

                # for k in range(len(empires[better_empire_ind].colonies_position)):
                #     x_i = empires[better_empire_ind].colonies_position[k]
                #     empires[better_empire_ind].colonies_fitness[k] = test_function.test_function(x_i, ProblemParam.test_function_index, ProblemParam.dimention)

                empires[better_empire_ind].total_fitness = empires[better_empire_ind].imperialist_fitness + AlgorithmParam.zeta * np.mean(empires[better_empire_ind].colonies_fitness)

                empires = np.delete(empires, [worse_empire_ind])

                return empires

    return empires


def imperialistic_competition(empires):
    if np.random.uniform(0, 1) > 0.11:
        return empires
    if len(empires) <= 1:
        return empires

    total_fit = np.zeros(len(empires))
    for i in range(len(empires)):
        total_fit[i] = empires[i].total_fitness

    max_total_fit = np.max(total_fit)
    ind_weakest_empire = np.argmax(total_fit)
    total_powers = max_total_fit - total_fit
    prossesion_probability = total_powers / np.sum(total_powers)

    selected_empire_ind = select_empire(prossesion_probability)

    num_colonies = len(empires[ind_weakest_empire].colonies_fitness)
    jj = int(np.ceil(np.random.uniform(0, 1) * num_colonies))

    if jj == num_colonies:
        jj = jj - 1

    if empires[ind_weakest_empire].colonies_position.size == 0:
        to_vstack = [empires[selected_empire_ind].colonies_position,
                     empires[ind_weakest_empire].colonies_position]
    else:
        to_vstack = [empires[selected_empire_ind].colonies_position,
                     empires[ind_weakest_empire].colonies_position[jj]]

    if (empires[selected_empire_ind].colonies_position.size != 0) or (empires[ind_weakest_empire].colonies_position.size != 0):
        empires[selected_empire_ind].colonies_position = np.vstack(tuple(filter(lambda x: x.size != 0, to_vstack)))

    if empires[ind_weakest_empire].colonies_fitness.size == 0:
        to_vstack = [empires[selected_empire_ind].colonies_fitness,
                     empires[ind_weakest_empire].colonies_fitness]
    else:
        to_vstack = [empires[selected_empire_ind].colonies_fitness,
                     empires[ind_weakest_empire].colonies_fitness[jj]]

    if (empires[selected_empire_ind].colonies_fitness.size != 0) or (
        empires[ind_weakest_empire].colonies_fitness.size != 0):
        empires[selected_empire_ind].colonies_fitness = np.hstack(tuple(filter(lambda x: x.size != 0, to_vstack)))
    del to_vstack

    empires[ind_weakest_empire].colonies_position = np.delete(empires[ind_weakest_empire].colonies_position, [jj], axis=0)
    empires[ind_weakest_empire].colonies_fitness = np.delete(empires[ind_weakest_empire].colonies_fitness, [jj])

    # крах самой слабой империи без колоний
    num_colonies = len(empires[ind_weakest_empire].colonies_fitness)
    if num_colonies <= 1:
        if (empires[selected_empire_ind].colonies_position.size != 0) or (empires[ind_weakest_empire].colonies_position.size != 0):
            empires[selected_empire_ind].colonies_position = np.vstack((empires[selected_empire_ind].colonies_position,
                                                                        empires[ind_weakest_empire].imperialist_position))

        if (empires[selected_empire_ind].colonies_position.size != 0) or (empires[ind_weakest_empire].colonies_position.size != 0):
            empires[selected_empire_ind].colonies_fitness = np.hstack((empires[selected_empire_ind].colonies_fitness,
                                                                        empires[ind_weakest_empire].imperialist_fitness))

        empires = np.delete(empires, [ind_weakest_empire])

    return empires


def select_empire(prossesion_probability):
    R = np.random.uniform(0, 1, prossesion_probability.shape)
    D = prossesion_probability - R
    index = np.argmax(D)
    return index


def optimization(empire, gamma, ProblemParam):
    """ Функция оптимизации положения империалиста (столицы).
        optization(empire, gamma, ProblemParam) -> empire
            Оптимизация осуществляется в небольшой окрестности.
            Окрестность (delta) расчитывается как gamma * best, 
        где gamma - настраеваемый параметр алгоритма (gamma < 1)
            best  - евклидово расстояние до ближайшей колонии.
            В случае, если у империалиста отсутствуют колонии область поиска определяется как gamma * gamma.
            Потенциальная позиция для перемещения выбирается случаынйм образом в данной окрестности:
        potential_position[i] = empire.imperialist_position[i] + np.random.uniform(-1, 1) * delta.
            Пригодность потенциального положения определяется путем расчета значения функции качества в этой точке,
        если оно [значение функции качества] меньше значения функции качества в текущей точке 
        происходит передвижение империалиста, в противном случае империалист не перемещается.
        --------------------------------------------------------
            Параметры:
        @param empire       - объект класса "Empires", 
                              необходимы поля - imperialist_position, colonies_position и imperialist_fitness
        @param gamma        - настраеваемый параметр алгоритма (0 < gamma < 1), число.
        @param ProblemParam - объект класса "ProblemParams"
            Возвращаемые значения:
        @return empire      - объект класса "Empires"
    """
    num_colonies = len(empire.colonies_position)

    if num_colonies == 0:
        best = gamma

    # поиск самой ближайшей колонии
    for i in range(num_colonies):
        distance_vector = empire.imperialist_position - empire.colonies_position[i]
        distance = np.linalg.norm(distance_vector)
        if i == 0:
            best = distance
        if best > distance:
            best = distance

    # Определение радиуса поисковой области
    delta = gamma * best

    potential_position = np.zeros(len(empire.imperialist_position))
    for i in range(len(empire.imperialist_position)):
        potential_position[i] = empire.imperialist_position[i] + np.random.uniform(-1, 1) * delta

    potential_fit = test_function.test_function(potential_position, ProblemParam.test_function_index, ProblemParam.dimention)

    if potential_fit < empire.imperialist_fitness:
        empire.imperialist_position = potential_position
        empire.imperialist_fitness = potential_fit

    return empire


def ICA(ProblemParam, AlgorithmParam, max_iter, min_flag, epsilon):
    low, up, dimension = test_function_range.get_range(ProblemParam.test_function_index)
    min_limit, max_limit, search_space_size = modifying_size_search_space(low, up, dimension)

    ProblemParam.max_limit_search = max_limit
    ProblemParam.min_limit_search = min_limit
    ProblemParam.search_space_size = search_space_size

    num_of_all_colonies = AlgorithmParam.num_of_countries - AlgorithmParam.num_of_initial_imperialists

    coord = np.empty((max_iter, AlgorithmParam.num_of_countries, dimension))

    # ----------Создание первоначальных империй----------
    initial_сountries = generate_new_country(AlgorithmParam.num_of_countries, min_limit, max_limit)
    # расчет приспособленности (стоимости) каждой страны
    initial_fitness_country = np.zeros(AlgorithmParam.num_of_countries)
    for i in range(AlgorithmParam.num_of_countries):
        initial_fitness_country[i] = test_function.test_function(initial_сountries[i], ProblemParam.test_function_index, dimension)

    # сортировка по убыванию
    ind_fitness_country = np.argsort(initial_fitness_country)[::-1]
    initial_fitness_country = np.sort(initial_fitness_country)[::-1]
    # initial_сountries = np.take(initial_сountries, ind_fitness_country, axis=0)
    initial_сountries_t = np.array([initial_сountries[ind_fitness_country[i]] for i in range(len(ind_fitness_country))])
    initial_сountries = initial_сountries_t

    empires = create_initial_empires(initial_сountries, initial_fitness_country, ProblemParam, AlgorithmParam)
    # ---------------------------------------------------

    # Главный цикл алгоритма
    min_fitness = np.zeros((max_iter, ))
    mean_fitness = np.zeros((max_iter, ))

    # best_chart = np.zeros(max_iter)
    # mean_chart = np.zeros(max_iter)
    best_chart = []
    mean_chart = []
    coord_x = np.zeros((max_iter, 1, dimension))
    stop_iteration = max_iter

    for i in range(max_iter):
        iteration = i + 1
        AlgorithmParam.revolution_rate = AlgorithmParam.revolution_rate * AlgorithmParam.damp_ratio
        remained = max_iter - iteration

        imperialist_fit = np.zeros(len(empires))
        for j in range(len(empires)):
            imperialist_fit[j] = empires[j].imperialist_fitness
        best = np.min(imperialist_fit)
        best_x = np.argmin(imperialist_fit)
        if iteration == 1:
            # лучшее значение функции
            func_best = best
            # лучший агент
            agent_best = empires[best_x].imperialist_position
        if best < func_best:
            func_best = best
            agent_best = empires[best_x].imperialist_position

        best_chart.append(func_best)
        coord_x[i][0] = agent_best
        mean_chart.append(np.mean(imperialist_fit))

        for j in range(len(empires)):
            empires[j] = optimization(empires[j], 0.1, ProblemParam)

            # Ассимиляция. Движение колоний к империалистам
            empires[j] = assimilate_colonies(empires[j], ProblemParam, AlgorithmParam)

            # Революция
            empires[j] = revolve_colonies(empires[j], ProblemParam, AlgorithmParam)

            # Расчет новой стоимости стран (приспособленности)
            fit = np.zeros(len(empires[j].colonies_position))
            for k in range(len(empires[j].colonies_position)):
                x_i = empires[j].colonies_position[k]
                fit[k] = test_function.test_function(x_i, ProblemParam.test_function_index, dimension)
            empires[j].colonies_fitness = fit

            empires[j] = prosses_empire(empires[j])

            # Вычисление общей стоимости (приспособленности) империи
            if empires[j].colonies_fitness.size == 0:
                empires[j].total_fitness = empires[j].imperialist_fitness
            else:
                empires[j].total_fitness = empires[j].imperialist_fitness + AlgorithmParam.zeta * np.mean(empires[j].colonies_fitness)

        # слияние империй
        empires = unite_simular_empires(empires, ProblemParam, AlgorithmParam)

        # империалистическая конкуренция
        empires = imperialistic_competition(empires)

        if (len(empires) == 1) and (AlgorithmParam.stop_if_just_one_empire):
            break
        # if iteration > 11:
        #     break_point = check_break(best_chart, epsilon)
        #     if break_point:
        #         stop_iteration = iteration
        #         break

    return func_best, agent_best, best_chart, mean_chart, coord_x, stop_iteration


def check_break(best_chart, epsilon, iter_for_break=10):
    e = pow(10, -10)
    k = 0
    res = True
    # for i in range(2, iter_for_break):
    #     res = res * (best_chart[-1] == best_chart[-i])
    # return res

    if np.abs(np.abs(best_chart[-2]) - np.abs(best_chart[-1])) <= epsilon:
        return True
    else:
        return False


def main():
    num_of_countries = 200  # общее количество стран
    num_of_initial_imperialists = 10  # начальное количество империалистов
    revolution_rate = 0.3
    assimilation_coefficient = 2  # коэффициент ассимиляции "beta"
    assimilation_angle_coefficient = 0.5  # угловой коэффициент ассимиляции "gama"
    zeta = 0.02
    damp_ratio = 0.99
    stop_if_just_one_empire = False
    uniting_threshold = 0.02
    zarib = 1.05
    alpha = 0.1

    epsilon = pow(10, -20)

    max_iter = 100
    min_flag = 1  # 1 - минимизация, 0 - максимизация
    # скорость изменения графика, установите значение в милисекундах
    rate_change_graph = 500

    # индекс функции в файле test_function
    f_index = 10

    low, up, dim = test_function_range.get_range(f_index)
    ProblemParam = ProblemParams.ProblemParams(dim, low, up, f_index, 0)
    AlgorithmParam = AlgorithmParams.AlgorithmParams(num_of_countries, num_of_initial_imperialists, zeta, revolution_rate, damp_ratio, assimilation_coefficient, stop_if_just_one_empire, uniting_threshold)

    print("Подождите идут вычисления...")

    func_best, agent_best, best_chart, mean_chart, coord_x, stop_iteration = ICA(ProblemParam, AlgorithmParam, max_iter, min_flag, epsilon)

    print("-------------------------------")
    print("Лучший результат: " + str(func_best))
    print("Лучшее решение (точка, агент): " + str(agent_best))
    print("Лучшие решения на каждой итерации: " + str(best_chart))
    print("Среднее решений на каждой итерации: " + str(mean_chart))
    print("-------------------------------")

    graph.graph_motion_points_3d(f_index, rate_change_graph, coord_x, max_iter)

    graph.graph_best_chart(best_chart)

    graph.print_graph(f_index, rate_change_graph, coord_x, max_iter)

    # graph.graph_with_arrow(f_index, rate_change_graph, coord_x, max_iter)


if __name__ == "__main__":
    main()
