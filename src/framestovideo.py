import cv2
from dataclasses import dataclass
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import os
import re
from scipy.interpolate import interp1d
from scipy.special import comb
from typing import Callable, Iterator, Sequence

from . import common
from .common import StrPath, StrKType


def smoothstep(order: int) -> Callable[[np.ndarray], np.ndarray]:
    def f(x: np.ndarray) -> np.ndarray:
        return x**(order + 1) * sum(comb(order + k, k) * comb(2*order + 1, order - k) * (-x)**k for k in range(order + 1))
    return f


def to_gray_from_alpha(img: np.ndarray) -> np.ndarray:
    return img[:,:,-1]


class _WorkingFrames:
    def __init__(self, source_dir: StrPath) -> None:
        self.source_dir = source_dir

        self.first: np.ndarray = None
        self.second: np.ndarray = None
        self.first_index: int = None
        self.second_index: int = None
    
    def read_image(self, frame_number: StrPath) -> np.ndarray:
        full_path = os.path.join(self.source_dir, common.frame_index_to_path(frame_number))
        image = cv2.imread(full_path, -1)
        alpha = np.float_(image[:,:,-1]/255)
        color = cv2.cvtColor(image[:,:,:3], cv2.COLOR_BGR2LAB)
        color[:, :, 0] = 255 - color[:, :, 0]
        color = np.float_(cv2.cvtColor(color, cv2.COLOR_LAB2BGR))
        
        for i in range(3):
            color[:, :, i] *= alpha

        return color
    
    def get(self, frame_number: float) -> np.ndarray:
        t = frame_number % 1
        first_index = int(frame_number)
        
        if first_index != self.first_index:
            if first_index == self.second_index:
                self.first = self.second
                self.second = self.read_image(first_index + 1)
            else:
                self.first = self.read_image(first_index)
                self.second = self.read_image(first_index + 1)
            
            self.first_index = first_index
            self.second_index = first_index + 1
        
        return (1 - t) * self.first + t * self.second


class Frame:
    fps: int = 30

    def __init__(self, *args) -> None:
        if len(args) == 1:
            self._total_subseconds, = args
        elif len(args) == 2:
            seconds, subseconds = args
            self._total_subseconds = self.fps * seconds + subseconds
        elif len(args) == 3:
            minutes, seconds, subseconds = args
            self._total_subseconds = self.fps * (60 * minutes + seconds) + subseconds
        else:
            raise ValueError(f'args length is {len(args)}, should be 1 or 2 or 3')

    def __repr__(self) -> str:
        return f'Frame({self.minutes}:{self.seconds:02d}:{self.subseconds:02d}, fps={self.fps})'

    @property
    def total_subseconds(self) -> int:
        return self._total_subseconds

    @property
    def subseconds(self) -> int:
        return self._total_subseconds % self.fps
    
    @property
    def seconds(self) -> int:
        return (self._total_subseconds // self.fps) % 60
    
    @property
    def minutes(self) -> int:
        return self._total_subseconds // (self.fps * 60)


class FramesToVideo:
    def __init__(self, width: int = 1920, height: int = 1080, source_dir: StrPath = 'frames', threads: int = 8, fps: int = 30, frames_range: Sequence[int] | None = None) -> None:
        self.width = width
        self.height = height
        self.source_dir = source_dir
        self.threads = threads
        self.fps = fps
        self.frames_range = frames_range

        self.position_keyframes: dict[int, int] = {}
        self.time_keyframes: dict[Frame, float] = {}
        self.frames: np.ndarray = None

    def keyframes(self, ktype: StrKType) -> dict[int, int] | dict[Frame, float]:
        if ktype == 'time':
            return {k.total_subseconds: v for k, v in self.time_keyframes.items()}
        elif ktype == 'position':
            return self.position_keyframes
        else:
            raise ValueError(f'ktype cannot be {ktype!r}')

    def read_keyframes_from_setting(self, filepath: StrPath, delta_subseconds: int = 0) -> Iterator[tuple[int, float]]:
        with open(filepath, 'r') as f:
            keyframes_string = re.search(r'KeyFrames = \{((.|\s)+?) \} \}\n', f.read())

        if not keyframes_string:
            return
        
        for line in keyframes_string.group(1).splitlines():
            line = line.lstrip('\t')

            frame_num = re.search(r'\[(.+?)\]', line)
            if not frame_num:
                continue
            frame_num = int(frame_num.group()[1:-1])

            value = re.search(r'] = { (.+?),', line)
            if not value:
                continue
            value = float(value.group()[5:-1])
                
            yield frame_num - delta_subseconds, value

    def delta_y_to_slices(self, delta_y: int, image_shape: tuple[int, int]) -> tuple[slice, ...]:
        image_height, image_width = image_shape

        if delta_y < 0:
            frame_y = (-delta_y, self.height)
            image_y = (0, self.height + delta_y)
        elif delta_y > image_height - self.height:
            frame_y = (0, image_height - self.height - delta_y)
            image_y = (image_height - delta_y, image_height)
        else:
            frame_y = (0, self.height)
            image_y = (delta_y, delta_y + self.height)
        
        if image_width <= self.width:
            frame_x = ((self.width - image_width) // 2, self.width - (self.width - image_width) // 2)
            image_x = (0, image_width)
        else:
            frame_x = (0, self.width)
            image_x = ((image_width - self.width) // 2, image_width - (image_width - self.width) // 2)

        return slice(*frame_y), slice(*frame_x), slice(*image_y), slice(*image_x)

    def frames_paths(self) -> Iterator[StrPath]:
        list_dir = os.listdir(self.source_dir)
        for frame in range(min(common.frame_path_to_index(path) for path in list_dir),
                           max(common.frame_path_to_index(path) for path in list_dir) + 1):
            frame_path = common.frame_index_to_path(frame)
            if frame_path not in list_dir:
                raise FileNotFoundError(f'{frame_path!r} does not exist in {self.source_dir!r}')
            
            full_path = os.path.join(self.source_dir, frame_path)
            if self.frames_range is not None:
                if frame in self.frames_range:
                    yield full_path
            else:
                yield full_path 

    def keyframes_interpolated(self, ktype: StrKType, kind: str) -> tuple[np.ndarray, np.ndarray]:
        x = np.array(list(self.keyframes(ktype).keys()))
        y = np.array(list(self.keyframes(ktype).values()))

        interp = interp1d(x, y, kind)
        x_interp = np.arange(0, max(x))
        y_interp = np.zeros(x_interp.shape[0])
        y_interp[min(x):] = interp(x_interp[min(x):])

        return x_interp, y_interp

    def read_time_keyframes(self, filepath: StrPath, subseconds_delta: int = 0) -> None:
        for subseconds, frame in self.read_keyframes_from_setting(filepath, subseconds_delta):
            self.time_keyframes[Frame(subseconds - subseconds_delta)] = frame
    
    def read_position_keyframes(self, filepath: StrPath, subseconds_delta: int = 0, time_interp_kind: str = 'linear') -> None:
        image_height = cv2.imread(os.path.join(self.source_dir, common.frame_index_to_path(1)), -1).shape[0]
        _, time_interpolated = self.keyframes_interpolated('time', time_interp_kind)

        for subseconds, relative_delta_y in self.read_keyframes_from_setting(filepath, subseconds_delta):
            frame = round(time_interpolated[subseconds])
            delta_y = round(image_height * relative_delta_y)
            self.position_keyframes[frame] = delta_y

    def add_keyframes(self, pos_keyframes: dict[int, int], time_keyframes: dict[int, int]) -> None:
        self.position_keyframes.update(pos_keyframes)
        self.time_keyframes.update(time_keyframes)

    def show_keyframes_graph(self, ktype: StrKType, /, kind: str = 'quadratic') -> None:
        x_interp, y_interp = self.keyframes_interpolated(ktype, kind)

        plt.plot(x_interp, y_interp)
        plt.plot(self.keyframes(ktype).keys(), self.keyframes(ktype).values(), 'o')
        if ktype == 'position':
            plt.xlabel('frame')
            plt.ylabel('Δy')
        else:
            plt.xlabel('subseconds')
            plt.ylabel('frame')
        plt.show()

    def convert(self, dtype: str = 'uint8', kind: str = 'quadratic') -> None:
        time_x_interp, time_y_interp = self.keyframes_interpolated('time', kind)
        length_subseconds = time_x_interp[np.argmin(np.abs(time_y_interp - max(self.frames_range)))]

        _, pos_y_interp = self.keyframes_interpolated('position', kind)
        length_frames = sum(1 for _ in self.frames_paths())
        self.frames = np.empty((length_subseconds, self.height, self.width, 3), dtype=np.dtype(dtype))

        working_frames = _WorkingFrames(self.source_dir)
        for subsecond, frame_number in zip(time_x_interp[:length_subseconds], time_y_interp[:length_subseconds]):
            t = frame_number % 1
            delta_y = (1 - t) * pos_y_interp[int(frame_number)] + t * pos_y_interp[int(frame_number) + 1]

            image = working_frames.get(frame_number)
            frame = np.zeros((self.height, self.width, 3), image.dtype)

            frame_y, frame_x, image_y, image_x = self.delta_y_to_slices(int(delta_y), image.shape[:2])
            subpixel_translation_matrix = np.array([
                [1, 0, 0],
                [0, 1, -delta_y % 1]
            ])

            frame[frame_y, frame_x] = image[image_y, image_x]
            frame = cv2.warpAffine(src=frame, M=subpixel_translation_matrix, dsize=(self.width, self.height))  # subpixel translation

            print(f'\r{subsecond}/{length_subseconds}', end='')

            self.frames[subsecond] = np.uint8(frame)
    
    def render_video(self, filepath: StrPath = 'out.mp4', fourcc: str = 'mp4v') -> None:
        videoout = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*fourcc), self.fps, (self.width, self.height), True)
        for i, frame in enumerate(self.frames):
            videoout.write(frame)
            print(f'\r{i}/{self.frames.shape[0]} ', end='')
        videoout.release()

if __name__ == '__main__':
    framestovideo = FramesToVideo()
