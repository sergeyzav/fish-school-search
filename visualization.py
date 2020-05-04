import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np


class Visualization:
    def __init__(self, history:dict, timeout=300, line_count=50, point_count=1000):
        self.history = history
        self.timeout = timeout
        self.line_count = line_count
        self.point_count = point_count
        self.iter = 0
        if history.get("dim") != 2:
            raise Exception("dim !== 2")

    def fitness(self, x, y):
        return self.history.get("fitness")([x, y])

    def start(self, saved=False,  filename="visual.mp4"):
        fig = plt.figure()

        ax = fig.add_subplot(projection='3d')
        ax.view_init(60, 48)

        self.draw_func(ax, self.line_count, self.point_count)

        pointer, = ax.plot3D([], [], marker="8", linestyle='None', markersize=2, color='red')

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        text = ax.text(0.05, 0.95, 0,  "", transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)

        def init():
            pointer.set_data([], [])
            return pointer,

        def update(fishes):
            x = np.array([fish.get("pos")[0] for fish in fishes], dtype=np.core.float64)
            y = np.array([fish.get("pos")[1] for fish in fishes], dtype=np.core.float64)
            z = np.array([fish.get("fitness_value") for fish in fishes], dtype=np.core.float64)
            pointer.set_data_3d(x, y, z)

            textstr = r'$max=%.10f$' % (max(z),)
            text.set_text(textstr)

            return pointer, text

        if not saved:
            ani = FuncAnimation(fig, update, frames=self.history.get("steps"),
                                init_func=init, blit=True, interval=self.timeout)
            plt.show()
        else:
            metadata = dict(title='Movie Test', artist='Matplotlib',
                            comment='Movie support!')
            movie_writer = FFMpegWriter(fps=5, metadata=metadata)
            with movie_writer.saving(fig, outfile=filename, dpi=1000):
                for step in self.history.get("steps"):
                    update(step)
                    movie_writer.grab_frame()

    def draw_func(self, ax, line_count=50, point_count=1000):
        x, y = np.meshgrid(self.get_x_line(point_count), self.get_y_line(point_count))
        z = self.fitness(x, y)

        ax.contour3D(x, y, z, line_count, cmap='binary')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    def get_x_line(self, point_count=1000):
        return np.linspace(self.history.get("lower")[0], self.history.get('higher')[0], point_count)

    def get_y_line(self, point_count=1000):
        return np.linspace(self.history.get("lower")[1], self.history.get('higher')[1], point_count)




