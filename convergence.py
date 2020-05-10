import matplotlib.pyplot as plt


class Convergence:
    def __init__(self, history: dict, max_value: float):
        self.history = history
        self.max_value = max_value

    def show(self, max_or_avg='avg'):
        fig = plt.figure()
        ax = fig.add_subplot()

        ax.set_xlabel('iteration')

        iterations = range(len(self.history.get('steps')))
        ax.plot(iterations, [self.max_value for _ in iterations], c="Green")

        if max_or_avg == 'max':
            max_fitness = [max(fish.get('fitness_value') for fish in fishes) for fishes in self.history.get('steps')]
            ax.set_ylabel('max')
            ax.plot(iterations, max_fitness, c="Red")
        else:
            ax.set_ylabel('avg')
            avg_fitness = [sum(fish.get('fitness_value') for fish in fishes) / len(fishes) for fishes in self.history.get('steps')]
            ax.plot(iterations, avg_fitness, c="Blue")

        plt.show()
