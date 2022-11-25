from audio2numpy import open_audio
from collections.abc import Iterator
import cv2
import numpy as np
from scipy.io.wavfile import write as write_wav
from scipy.ndimage import zoom, gaussian_filter
from scipy.signal import argrelmax

from .leftgoes import Progress, linmap, Colormap

NoteInfo = str | float


class Munim:
    def __init__(self, fps: int = 30, width: int = 1920, height: int = 1080, cmap: str = 'magma') -> None:
        self.fps = fps
        self.width = width
        self.height = height

        self.sample_rate: int = None

        self._audio: np.ndarray = None
        self._progress = Progress()

        self._cmap = Colormap.get(cmap)
    
    @property
    def frame_length(self) -> int:
        return round(self.sample_rate / self.fps)

    def cmap(self, x: float) -> np.ndarray:
        return

    def frames(self) -> Iterator[int]:
        for t0 in range(0, self._audio.shape[0] - self.frame_length, self.frame_length):
            yield t0

    def read_audio(self, filepath: str, start: float = 0, end: float = 1) -> None:
        data, self.sample_rate = open_audio(filepath)
        if len(data.shape) == 1:
            self._audio = data[round(start * data.shape[0]):round(end * data.shape[0]) - 1]
        elif len(data.shape) == 2:
            self._audio = sum(data[round(start * data.shape[0]):round(end * data.shape[0]) - 1, i] for i in range(data.shape[1])) / data.shape[1]

    def render_video(self, *args, **kwargs) -> None:
        return


class Spectrum(Munim):
    a4_hz: float = 440
    _note_offset: dict[str, int] = {'C': -9, 'D': -7, 'E': -5, 'F': -4,
                                    'G': -2, 'A': 0, 'B': 2}
    _accidental_offset: dict[str, int] = {'bb': -2, 'b': -1, '#': 1, 'x': 2}  

    def __init__(self, fps: int = 30, width: int = 1920, height: int = 1080, cmap: str = 'magma') -> None:
        super().__init__(fps, width, height, cmap)

        self._transformed: np.ndarray = None

    def hz(self, note: NoteInfo) -> float:
        if isinstance(note, float): return note
        elif not isinstance(note, str): raise ValueError(f'cannot get frequency from {note!r}')

        pitch = self._note_offset[note[0].upper()]
        if len(note) == 1:
            return self.a4_hz * 2**(pitch/12)
        elif len(note) == 2:
            if note[1] in self._accidental_offset:
                return self.a4_hz * 2**((pitch + self._accidental_offset[note[1]])/12)
            else:
                return self.a4_hz * 2**(pitch/12 + int(note[1]) - 4)
        else:
            accidental, octave = note[1:-1], note[-1]
            return self.a4_hz * 2**((pitch + self._accidental_offset[accidental])/12 + int(octave) - 4)

    def cmap(self, x: float) -> np.ndarray:
        return self._cmap(x)[:, :3][:, ::-1]

    def normalized(self, gamma: float, highlight_clip: float, invert: bool):
        freq = self._transformed - self._transformed.min()
        freq = np.clip(highlight_clip * freq/freq.max(), 0, 1)**gamma
        return 1 - freq if invert else freq

    def render_video(self, filepath: str, fourcc: str = 'mp4v', gamma: float = 1, highlight_clip: float = 1, invert: bool = False, show_frames: bool = False) -> None:
        self._progress.start()
        videoout = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*fourcc), self.fps, (self.width, self.height), True)

        frames = self.normalized(gamma, highlight_clip, invert)
        for i, frame_data in enumerate(frames):
            zoomed = frame_data if frame_data.shape[0] == self.width else zoom(frame_data, self.width / frame_data.shape[0])
            colored = self.cmap(zoomed)
            frame = np.uint8(255 * np.tile(colored, (self.height, 1, 1)))

            if show_frames:
                cv2.imshow(f'frame {i}', frame)
                cv2.waitKey(0)

            videoout.write(frame)
            self._progress.string((i + 1)/len(frames))
        videoout.release()
    
    def show_spectrum(self, gamma: float = 1) -> None:
        cv2.imshow(self.__class__.__name__, np.uint8(255 * (self._transformed/self._transformed.max())**gamma))
        cv2.waitKey(0)

    def save_spectrum(self, filepath: str, gamma: float = 1, dtype: np.dtype | str = 'uint16') -> None:
        cv2.imwrite(filepath, (np.iinfo(dtype).max * (self._transformed/self._transformed.max())**gamma).astype(dtype))

    def transform(self, *args, **kwargs) -> None:
        return


class STFT(Spectrum):
    def __init__(self, fps: int = 30, width: int = 1920, height: int = 1080, cmap: str = 'magma') -> None:
        super().__init__(fps, width, height, cmap)

    def transform(self, lowest: NoteInfo = 'A0', highest: NoteInfo = 'C8', snippet_frames_num: float = 10) -> None:
        min_hz, max_hz = self.hz(lowest), self.hz(highest)
        length = round(snippet_frames_num * self.frame_length)

        stft_x = np.fft.fftfreq(length, 1 / self.sample_rate)
        indices = np.where(np.logical_and(min_hz <= stft_x, stft_x <= max_hz))
        freq_sample_num = indices[0].shape[0]

        self._transformed = np.empty((int(self._audio.shape[0] / self.frame_length), freq_sample_num))
        for i, t0 in enumerate(self.frames()):
            snippet = self._audio[t0:t0 + length]
            stft_y = np.fft.fft(snippet)

            self._transformed[i, :] = np.abs(stft_y[indices])


class Wavelet(Spectrum):
    def __init__(self, fps: int = 30, width: int = 1920, height: int = 1080, cmap: str = 'magma') -> None:
        super().__init__(fps, width, height, cmap)
    
    @staticmethod
    def gaussian(x: np.ndarray, sigma: float) -> np.ndarray:
        return np.exp(-x**2/(2 * sigma**2))

    def psi(self, x: np.ndarray, f: float, sigma: float) -> np.ndarray:
        return

    def transform(self, lowest: NoteInfo = 'A0', highest: NoteInfo = 'C8', *, sigma_seconds: float = 0.1, radius: float = 4) -> None:
        min_hz, max_hz = self.hz(lowest), self.hz(highest)
        
        sigma = round(self.sample_rate * sigma_seconds)  # sigma in samples
        length = round(2*radius*sigma)

        freqs = 2**np.linspace(np.log2(min_hz), np.log2(max_hz), self.width)
        x = np.arange(-radius*sigma, radius*sigma)

        self._progress.start()
        self._transformed = np.empty((int(self._audio.shape[0] / self.frame_length) + 1, freqs.shape[0]))
        for j, f in enumerate(freqs):
            wavelet = self.psi(x, f/self.sample_rate, sigma)
            for i, t0 in enumerate(self.frames()):
                if t0 + length > self._audio.shape[0] - 1:
                    snippet = self._audio[t0:]
                    self._transformed[i, j] = np.abs(np.dot(wavelet[:snippet.shape[0]], snippet))
                else:
                    snippet = self._audio[t0:t0 + length]
                    self._transformed[i, j] = np.abs(np.dot(wavelet, snippet))
                self._progress.string(((i + 1)/self._transformed.shape[0] + j)/freqs.shape[0])
        self._progress.string(1)


class Morlet(Wavelet):
    def __init__(self, fps: int = 30, width: int = 1920, height: int = 1080, cmap: str = 'magma') -> None:
        super().__init__(fps, width, height, cmap)
    
    def psi(self, x: np.ndarray, f: float, sigma: float) -> np.ndarray:
        return self.gaussian(x, sigma) * np.exp(1j * 2*np.pi*f * x)


class Custom(Wavelet):
    def __init__(self, fps: int = 30, width: int = 1920, height: int = 1080, cmap: str = 'magma') -> None:
        super().__init__(fps, width, height, cmap)
        self._calib: np.ndarray = None
    
    def psi(self, x: np.ndarray, f: float, sigma: float) -> np.ndarray:
        y = np.zeros(x.shape, dtype=np.complex_)
        for n, amp, phi in self._calib:
            y += amp * np.exp(1j * 2*np.pi * n*f * x + phi)
        return self.gaussian(x, sigma) * y

    def calibrate(self, filepath: str, base_hz: float, overtones: int = 25, radius_channels: int = 1, start: float = 0, end: float = 1) -> int:
        data, sample_rate = open_audio(filepath)
        if len(data.shape) == 1:
            data = data[round(start * data.shape[0]):round(end * data.shape[0]) - 1]
        elif len(data.shape) == 2:
            data = sum(data[round(start * data.shape[0]):round(end * data.shape[0]) - 1, i] for i in range(data.shape[1])) / data.shape[1]
        else:
            return
        
        y = np.fft.fft(data)
        x = np.fft.fftfreq(y.size, 1/sample_rate)[:y.size // 2]
        y = y[:y.size // 2]
        
        self._calib = np.empty((overtones, 3))
        maxima, = argrelmax(y[:50000], order=1000)
        for i, maximum in enumerate(maxima):
            if i == overtones: break
            index_min = maximum - radius_channels + 1
            index_max = maximum + radius_channels
            self._calib[i, 0] = np.sum(x[index_min:index_max]/base_hz)/(index_max - index_min)
            self._calib[i, 1] = np.sum(np.abs(y[index_min:index_max]))/(index_max - index_min)
            self._calib[i, 2] = np.sum(np.angle(y[index_min:index_max]))/(index_max - index_min)
        return len(maxima)
        
    def hear_tone(self, filepath: str, base_hz: float, length_seconds: float = 2, sample_rate: int | None = None) -> None:
        if sample_rate is None: sample_rate = self.sample_rate
        x = np.linspace(0, length_seconds, round(sample_rate * length_seconds))
        y = np.zeros(x.shape)

        for n, amp, phi in self._calib:
            y += (amp * np.exp(1j * 2*np.pi * n * base_hz * x + phi)).real
        y /= np.abs(y).max()

        write_wav(filepath, sample_rate, np.int16(32767 * y))


class Oscillate(Munim):
    def __init__(self, fps: int = 30, width: int = 1080, cmap: str = 'magma') -> None:
        super().__init__(fps, width, -1, cmap)
    
    def cmap(self, x: np.ndarray) -> np.ndarray:
        return self._cmap(x)[:,:,:3][:,:,::-1]

    def read_audio(self, filepath: str, start: float = 0, end: float = 1) -> None:
        data, self.sample_rate = open_audio(filepath)

        if len(data.shape) in (1, 2):
            self._audio = data[round(start * data.shape[0]):round(end * data.shape[0]) - 1]/np.abs(data).max()

    def render_video(self, filepath: str, fourcc: str = 'mp4v', darken: float = 0.05) -> None:
        self._progress.start()
        videoout = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*fourcc), self.fps, (self.width, self.width), True)
        length = self.frame_length

        frame = np.zeros((self.width, self.width))
        if len(self._audio.shape) == 1:
            pass
        else:
            for i, (right, left) in enumerate(self._audio):
                x = int(linmap(right, (-1, 1), (0, self.width - 1)))
                y = int(linmap(left, (-1, 1), (0, self.width - 1)))

                frame[y, x] += (1 - (i % length) / length) * 0.07
                if i % length == 0:
                    frame *= 1 - darken
                    videoout.write(np.uint8(255 * self.cmap(np.clip(frame, 0, 1))))
                    self._progress.string(i/self._audio.shape[0])
            
        self._progress.string(1)
        videoout.release()
