import numpy as np
import math
from typing import Iterable
from typing import Callable
import time


class FishSchoolSearch:
    class __Fish:
        def __init__(self, swim_start_point: np.core.ndarray, weight: np.core.float64):
            self.swim_position = np.array([swim_start_point for i in range(4)])
            self.weight = weight
            self.individual_delta_point = np.zeros(swim_start_point.size, dtype=np.core.float64)
            self.individual_delta_fitness = 0

    def history(self):
        return {
            "steps": self.__history,
            "dim": self.__func_dimension_size,
            "fitness": self.fitness,
            "lower": self.__lower_bound_point,
            'higher': self.__higher_bound_point,
        }

    def log(self):
        fishes = [{
            "pos": fish.swim_position[0].copy(),
            "fitness_value": self.fitness(fish.swim_position[0]),
            "weight": fish.weight,
        } for fish in self.__fishes]

        self.__history.append(fishes)

    def __init__(
            self,
            lower_bound_point: Iterable[int],
            higher_bound_point: Iterable[int],
            population_size: int,
            iteration_count: int,
            individual_step_start: float,
            individual_step_final: float,
            weight_scale: float,
            func: Callable[[Iterable[float]], float]
    ):
        time_ = time.time()
        self.__lower_bound_point = np.array(lower_bound_point, dtype=np.core.float64)
        self.__higher_bound_point = np.array(higher_bound_point, dtype=np.core.float64)
        self.__population_size = np.core.int64(population_size)
        self.__iteration_count = np.core.int64(iteration_count)
        self.__individual_step_start = np.core.float64(individual_step_start)
        self.__individual_step_final = np.core.float64(individual_step_final)
        self.__weight_scale = np.core.float64(weight_scale)
        self.__func_dimension_size = self.__lower_bound_point.size
        self.__fitness_func = func
        self.__history = []

        if self.__lower_bound_point.size != self.__higher_bound_point.size:
            raise BaseException("fhfg")  # todo

        self.__fishes = self.__init_population()

        self.instinct_sum_weight_new = self.__population_size * self.__weight_scale / 2.
        self.individual_step = self.__individual_step_start

        for iteration in range(self.__iteration_count):
            for i in range(self.__population_size):
                self.__fishes[i] = self.__individual_movement(self.__fishes[i], self.individual_step)

            self.max_individual_delta_fitness = self.__search_max_individual_delta_fitness()

            for i in range(self.__population_size):
                self.__fishes[i] = self.__feeding(self.__fishes[i])

            self.instinct_average = self.__get_instinct_average_weighted()

            for i in range(self.__population_size):
                self.__fishes[i] = self.__instinct_swim(self.__fishes[i], self.instinct_average)

            self.barycentre = self.__search_barycentre()

            self.will_step = 2 * self.individual_step

            self.instinct_sum_weight_old = self.instinct_sum_weight_new
            self.instinct_sum_weight_new = self.__sum_weight()

            for i in range(self.__population_size):
                self.__fishes[i] = self.__collective_swim(self.__fishes[i])

            for i in range(self.__population_size):
                self.__fishes[i].swim_position[0] = self.__fishes[i].swim_position[3]

            self.individual_step -= (self.__individual_step_start
                                     - self.__individual_step_final) / self.__iteration_count

            self.log()

        self.time = time.time() - time_

    def max(self):
        max_ = 0
        pos_ = np.zeros(self.__func_dimension_size, dtype=np.core.float64)
        for fish in self.__fishes:
            if max_ <= self.fitness(fish.swim_position[0]):
                max_ = self.fitness(fish.swim_position[0])
                pos_ = fish.swim_position[0].copy()

        return max_, pos_

    def __init_population(self) -> [__Fish]:
        r"""
        Инициализирует популяцию рыбок
        :return:
        :rtype:
        """
        return np.array([self.__generate_new_fish() for i in range(self.__population_size)], dtype=self.__Fish)

    def __generate_new_fish(self) -> __Fish:
        r"""
        Генерирует рыбку
        :return:
        :rtype:
        """
        fish_position = (self.__higher_bound_point - self.__lower_bound_point) \
                        * np.random.random_sample(self.__func_dimension_size) + self.__lower_bound_point

        weight = np.core.float64(self.__weight_scale / 2)

        return self.__Fish(fish_position, weight)

    def __individual_movement(self, fish: __Fish, individual_step: np.core.float64) -> __Fish:
        r"""
        Индивидуальное плаванье рыбки
        :param fish:
        :type fish:
        :param individual_step:
        :type individual_step:
        :return:
        :rtype:
        """
        fish.individual_delta_point = np.zeros(self.__func_dimension_size, dtype=np.core.float64)
        fish.individual_delta_fitness = 0
        fish.swim_position[1] = fish.swim_position[0]

        new_pos = fish.swim_position[0] \
                  + np.random.uniform(-1, 1, self.__func_dimension_size) * individual_step

        if not self.__fish_in_shape(new_pos):
            return fish

        individual_delta_fitness = self.fitness(new_pos) - self.fitness(fish.swim_position[0])

        if individual_delta_fitness < 0:
            return fish

        fish.swim_position[1] = new_pos
        fish.individual_delta_point = fish.swim_position[1] - fish.swim_position[0]
        fish.individual_delta_fitness = self.fitness(fish.swim_position[1]) - self.fitness(fish.swim_position[0])

        return fish

    def __fish_in_shape(self, point: np.core.ndarray) -> bool:
        r"""
        Возращает True если точка внутри области, иначе False
        :param point:
        :type point:
        :return:
        :rtype:
        """
        for i in range(self.__func_dimension_size):
            if point[i] < self.__lower_bound_point[i] \
                    or point[i] > self.__higher_bound_point[i]:
                return False

        return True

    def fitness(self, x: np.core.ndarray) -> np.core.float64:
        r"""
        Фитнесс функция
        :param x:
        :type x:
        :return:
        :rtype:
        """
        return np.core.float64(self.__fitness_func(x))

    def __feeding(self, fish: __Fish) -> __Fish:
        r"""
        Кормление рыбки
        :param fish:
        :type fish:
        :return:
        :rtype:
        """
        if self.max_individual_delta_fitness != 0:
            fish.weight += fish.individual_delta_fitness / self.max_individual_delta_fitness

        if fish.weight < 1:
            fish.weight = 1

        if fish.weight > self.__weight_scale:
            fish.weight = self.__weight_scale

        return fish

    def __search_max_individual_delta_fitness(self) -> np.core.float64:
        r"""
        Поиск максимального значения дельта-функции
        :return:
        :rtype:
        """
        return np.core.float64(max(fish.individual_delta_fitness for fish in self.__fishes))

    def __get_instinct_average_weighted(self) -> np.core.ndarray:
        r"""
        Подсчет среднего взвешенного индивидуальных движений

        :return:
        :rtype:
        """
        numerator = np.zeros(self.__func_dimension_size, dtype=np.core.float64)
        denominator = 0

        for fish in self.__fishes:
            numerator += fish.individual_delta_point * fish.individual_delta_fitness
            denominator += fish.individual_delta_fitness

        if denominator == 0:
            return np.zeros(self.__func_dimension_size, dtype=np.core.float64)

        return numerator / denominator

    def __instinct_swim(self, fish: __Fish, instinct_average: np.core.ndarray) -> __Fish:
        r"""
        Инстинкитвное плавание рыбки
        :param fish:
        :type fish:
        :param instinct_average:
        :type instinct_average:
        :return:
        :rtype:
        """
        fish.swim_position[2] = fish.swim_position[1] + instinct_average

        if not self.__fish_in_shape(fish.swim_position[2]):
            fish.swim_position[2] = fish.swim_position[1]

        return fish

    def __search_barycentre(self) -> np.core.ndarray:
        r"""
        Поиск центра тяжести системы
        :return:
        :rtype:
        """
        numerator = np.zeros(self.__func_dimension_size, dtype=np.core.float64)
        denominator = 0

        for fish in self.__fishes:
            numerator += fish.swim_position[2] * fish.weight
            denominator += fish.weight

        if denominator == 0:
            return np.zeros(self.__func_dimension_size, dtype=np.core.float64)

        return numerator / denominator

    def __sum_weight(self) -> np.core.ndarray:
        r"""
        Суммарный вес всех рыб
        :return:
        :rtype:
        """
        return sum(fish.weight for fish in self.__fishes)

    def __collective_swim(self, fish: __Fish) -> __Fish:
        r"""
        Коллективно-волевое плавание рыбки
        :param fish:
        :type fish:
        :return:
        :rtype:
        """
        distance = self.__distance(fish.swim_position[2], self.barycentre)

        if distance == 0:
            fish.swim_position[3] = fish.swim_position[2]
            return fish

        diff = (fish.swim_position[2] - self.barycentre) \
               * self.will_step * np.random.uniform(0, 1, 1) / distance

        if self.instinct_sum_weight_new > self.instinct_sum_weight_old:
            fish.swim_position[3] = fish.swim_position[2] - diff
        else:
            fish.swim_position[3] = fish.swim_position[2] + diff

        for i in range(self.__func_dimension_size):
            if fish.swim_position[3][i] < self.__lower_bound_point[i]:
                fish.swim_position[3][i] = self.__lower_bound_point[i]

            if fish.swim_position[3][i] > self.__higher_bound_point[i]:
                fish.swim_position[3][i] = self.__higher_bound_point[i]

        return fish

    @staticmethod
    def __distance(fist_point: np.core.ndarray, second_point: np.core.ndarray) -> np.core.float64:
        r"""
        Евклидовое растояниу между двумя точками
        :param fist_point:
        :type fist_point:
        :param second_point:
        :type second_point:
        :return:
        :rtype:
        """
        diff = second_point - fist_point
        diff = diff * diff
        diff = diff.sum()
        return np.core.float64(math.sqrt(diff))
