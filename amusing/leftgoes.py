from datetime import datetime, timedelta
from matplotlib.cm import get_cmap
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import time

T = float | np.ndarray


class Progress:
    def __init__(self, fmt: str = '[{dtime:%H:%M:%S}] {bar} | {percent:.2f}% | {elapsed} | {remaining}', bar_length: int = 30) -> None:
        self.fmt = fmt
        self._checkpoints: list[float] = []
        self._start: float = None

        def progressbar(progress: float) -> str:
            quarters = '_░▒▓█'
            done = int(progress * bar_length)
            return (done * '█' + quarters[round(4 * (bar_length * progress - done))] + int((1 - progress) * bar_length) * '_')[:bar_length]
        self.progressbar = progressbar

    def start(self) -> float:
        self._start = time.perf_counter()
        return self._start
    
    def checkpoint(self) -> float:
        perf = time.perf_counter()
        self._checkpoints.append(perf)
        return perf

    def string(self, progress: float, use_checkpoint: int = -1, print_string: bool = True, *, prefix: str = '', suffix: str = '', **kwargs) -> str:
        delta = time.perf_counter() - (self._start if use_checkpoint == -1 else self._checkpoints[use_checkpoint])
        string = self.fmt.format(bar=self.progressbar(progress),
                                 percent=100 * progress,
                                 dtime=datetime.now(),
                                 elapsed=str(timedelta(seconds=round(delta))),
                                 remaining=str(timedelta(seconds=0 if progress == 0.0 else round(delta/progress))),
                                 **kwargs)
        if prefix != '':
            string = f'{prefix} | ' + string
        if suffix != '':
            string += f' | {suffix}'

        if print_string:
            print(f'\r{string}', end="\n" if progress == 1.0 else "")
        return string


class Colormap:
    @staticmethod
    def cold(n: int = 256, gamma: float = 0.6):
        cdict = {'red': [[0.0, 0.0, 0.0],
                         [0.2, 0.043, 0.043],
                         [0.4, 0.023, 0.023],
                         [0.5, 0.074, 0.074],
                         [0.6, 0.277, 0.277],
                         [1.0, 0.871, 0.871]],
               'green': [[0.0, 0.0, 0.0],
                         [0.2, 0.094, 0.094],
                         [0.4, 0.156, 0.156],
                         [0.5, 0.387, 0.387],
                         [0.6, 0.707, 0.707],
                         [1.0, 0.961, 0.961]],
                'blue': [[0.0, 0.0, 0.0],
                         [0.2, 0.125, 0.125],
                         [0.4, 0.238, 0.238],
                         [0.5, 0.871, 0.871],
                         [0.6, 1.0, 1.0],
                         [1.0, 1.0, 1.0]]}
        return LinearSegmentedColormap('cold', cdict, N=n, gamma=gamma)
    
    @classmethod
    def get(cls, name: str, n: int = 256):
        if name in cls.__dict__:
            return cls.__getattribute__(cls, name)(n)
        else:
            return get_cmap(name)

def linmap(x: T, from_range: tuple[T, T], to_range: tuple[T, T]) -> T:
    return to_range[0] + (to_range[1] - to_range[0]) * (x - from_range[0])/(from_range[1] - from_range[0])
