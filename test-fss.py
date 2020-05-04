from fss.fss import FishSchoolSearch
import numpy as np
import matplotlib.pyplot as plt
import math
import visualization



def test():
    pass


def test_1():
    fss = FishSchoolSearch(
        lower_bound_point=[-10, -10],
        higher_bound_point=[10, 10],
        population_size=50,
        iteration_count=100,
        individual_step_start=2,
        individual_step_final=0.01,
        weight_scale=50,
        func=lambda x: -0.1 * (x[0] * x[0] + x[1] * x[1]) + 20,
    )

    v = visualization.Visualization(fss.history())
    v.start()


def test_2():
    fss = FishSchoolSearch(
        lower_bound_point=[-100, -100],
        higher_bound_point=[100, 100],
        population_size=50,
        iteration_count=100,
        individual_step_start=10,
        individual_step_final=0.01,
        weight_scale=50,
        func=lambda x: 10 * (np.sin(0.1*x[0]) + np.sin(0.1*x[1])) + 20,
    )

    v = visualization.Visualization(fss.history())
    v.start()

if __name__ == '__main__':
    test_2()
