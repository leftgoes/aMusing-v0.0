import cv2
from noise import pnoise3
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import random
from typing import Self
from dataclasses import dataclass


@dataclass(order=True)
class Vector3:
    x: float
    y: float
    z: float

    def __repr__(self) -> str:
        return str((round(self.x, 2), round(self.y, 2), round(self.z, 2)))
    
    @property
    def xyz(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z

@dataclass
class Vector2:
    x: float
    y: float

class Particle:
    mag_offset: float = 420.69
    mean_halflife: float = 10
    
    def __init__(self, width: int, height: int, z: float = 0, speed: float = 1) -> None:
        self.width = width
        self.height = height

        self.previous: Vector3 = None

        x, y = self.random_xy()
        self.pos = Vector3(x, y, z)
        self.decay_constant = np.log(2) / random.expovariate(self.mean_halflife)

        self.speed = speed
        
    def __repr__(self) -> str:
        return f'Particle{repr(self.pos)}'

    @classmethod
    def random(cls, *args, **kwargs) -> Self:
        return cls(*args, **kwargs)

    @classmethod
    def randoms(cls, num: int, *args, **kwargs) -> list[Self]:
        return [cls.random(*args, **kwargs) for _ in range(num)]

    def random_xy(self) -> tuple[float, float]:
        return (self.width - 1) * random.random(), (self.height - 1) * random.random()

    def update(self, scale: float, delta_theta: float = 0) -> bool:
        theta = 3 * np.pi * (pnoise3(self.pos.x / scale, self.pos.y / scale, self.pos.z / scale) + 1) + delta_theta
        mag = 1 # pnoise3(self.pos.x / scale + self.mag_offset, self.pos.y / scale + self.mag_offset, self.pos.z / scale + self.mag_offset)

        self.previous = self.pos

        self.pos.x = (self.pos.x + self.speed * mag * np.cos(theta)) % self.width
        self.pos.y = (self.pos.y + self.speed * mag * np.sin(theta)) % self.height

        self.pos.z += self.speed

        if random.random() < self.decay_constant:
            self.respawn()
    
    def respawn(self) -> None:
        self.previous: Vector3 = None
        self.pos.x, self.pos.y = self.random_xy()
        self.decay_constant = random.expovariate(self.mean_halflife)


class Merlin:
    def __init__(self, width: int = 1920, height: int = 1080, scale: float = 30, speed: float = 1) -> None:
        self.width = width
        self.height = height
        self.scale = scale
        self.speed = speed

        self.image = np.ndarray

    def draw_particle(self, particle: Particle) -> None:
        self.image[int(particle.pos.y), int(particle.pos.x)] += 1
    
    def show_image(self, delay_ms: int = 0, winname: str = 'self.image', darken: float = 1) -> None:
        cv2.imshow(winname, self.image / darken)
        cv2.waitKey(delay_ms)

    def render_video(self, filepath: str, num_frames: int = 1000, num_particles: int = 10000, fps: int = 30, fourcc: str = 'mp4v', show_frames: bool = False):
        particles = Particle.randoms(num_particles, width=self.width, height=self.height, speed=self.speed)
        
        videoout = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*fourcc), fps, (self.width, self.height), False)
        self.image = np.zeros((self.height, self.width))
        for frame in range(num_frames):
            print(f'\r{frame}/{num_frames}', end='')
            for particle in particles:
                self.draw_particle(particle)
                particle.update(self.scale)
            self.image *= 0.98
            videoout.write((10 * self.image).astype(np.uint8))

            if show_frames:
                self.show_image(20, darken=50)
        videoout.release()

if __name__ == '__main__':
    merlin = Merlin(1920, 1080, 300, .5)
    merlin.render_video('part.mp4', num_frames=6000, fps=60, num_particles=100000)