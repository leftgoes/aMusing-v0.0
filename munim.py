from audio2numpy import open_audio
import cv2
from matplotlib.cm import get_cmap
import numpy as np
from scipy.ndimage import zoom

from leftgoes.utils import Progress

NoteInfo = str | float


class Munim:
    _a4_hz: float = 440
    _note_offset: dict[str, int] = {'C': -9, 'D': -7, 'E': -5, 'F': -4,
                                    'G': -2, 'A': 0, 'B': 2}
    _accidental_offset: dict[str, int] = {'bb': -2, 'b': -1, '#': 1, 'x': 2}

    def __init__(self, fps: int = 30, width: int = 1920, height: int = 1080, cmap: str = 'magma') -> None:
        self.fps = fps
        self.width = width
        self.height = height
        self.sample_rate: int = 0

        self._audio: np.ndarray = None
        self._progress = Progress()
        self._transformed: np.ndarray = None

        self.__cmap = get_cmap(cmap)

    @staticmethod
    def hz(note: NoteInfo) -> float:
        if isinstance(note, float): return note
        elif not isinstance(note, str): raise ValueError(f'cannot get frequency from {note!r}')

        pitch = Munim._note_offset[note[0].upper()]
        if len(note) == 1:
            return Munim._a4_hz * 2**(pitch/12)
        elif len(note) == 2:
            if note[1] in Munim._accidental_offset:
                return Munim._a4_hz * 2**((pitch + Munim._accidental_offset[note[1]])/12)
            else:
                return Munim._a4_hz * 2**(pitch/12 + int(note[1]) - 4)
        else:
            accidental, octave = note[1:-1], note[-1]
            return Munim._a4_hz * 2**((pitch + Munim._accidental_offset[accidental])/12 + int(octave) - 4)


    def cmap(self, x: float) -> np.ndarray:
        return self.__cmap(x)[:, :3][:, ::-1]

    def normalized(self, gamma: float, highlight_clip: float, invert: bool):
        freq = self._transformed - self._transformed.min()
        freq = np.clip(highlight_clip * freq/freq.max(), 0, 1)**gamma
        return 1 - freq if invert else freq
    
    def read_audio(self, filepath: str) -> None:
        data, self.sample_rate = open_audio(filepath)
        if len(data.shape) == 1:
            self._audio = data
        elif len(data.shape) == 2:
            self._audio = sum(data[:, i] for i in range(data.shape[1])) / data.shape[1]

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


class STFT(Munim):
    def __init__(self, fps: int = 30, width: int = 1920, height: int = 1080, cmap: str = 'magma') -> None:
        super().__init__(fps, width, height, cmap)

    def transform(self, lowest: NoteInfo = 'A0', highest: NoteInfo = 'C8', snippet_frames_num: float = 10) -> None:
        min_hz, max_hz = self.hz(lowest), self.hz(highest)
        frame_sample_num: int = round(self.sample_rate / self.fps)
        length = round(snippet_frames_num * frame_sample_num)

        stft_x = np.fft.fftfreq(length, 1 / self.sample_rate)
        indices = np.where(np.logical_and(min_hz <= stft_x, stft_x <= max_hz))
        freq_sample_num = indices[0].shape[0]

        self._transformed = np.empty((int(self._audio.shape[0] / frame_sample_num), freq_sample_num))
        for i, t0 in enumerate(np.arange(0, self._audio.shape[0] - frame_sample_num, frame_sample_num)):
            snippet = self._audio[t0:t0 + length]
            stft_y = np.fft.fft(snippet)

            self._transformed[i, :] = np.abs(stft_y[indices])


class Morlet(Munim):
    def __init__(self, fps: int = 30, width: int = 1920, height: int = 1080, cmap: str = 'magma') -> None:
        super().__init__(fps, width, height, cmap)
    
    def transform(self, lowest: NoteInfo = 'A0', highest: NoteInfo = 'C8', *, sigma_seconds: float = 0.1, radius: float = 4) -> None:
        min_hz, max_hz = self.hz(lowest), self.hz(highest)
        
        frame_sample_num: int = round(self.sample_rate / self.fps)
        sigma = round(self.sample_rate * sigma_seconds)  # sigma in samples
        length = round(2*radius*sigma)

        gaussian = lambda x: np.exp(-x**2/(2 * sigma**2))
        sin = lambda x, f: np.exp(1j * 2*np.pi*f * x)

        freqs = 2**np.linspace(np.log2(min_hz), np.log2(max_hz), self.width)
        x = np.arange(-radius*sigma, radius*sigma)

        self._progress.start()
        self._transformed = np.empty((int(self._audio.shape[0] / frame_sample_num) + 1, freqs.shape[0]))
        for j, f in enumerate(freqs):
            wavelet: np.ndarray = gaussian(x) * sin(x, f/self.sample_rate)
            for i, t0 in enumerate(np.arange(0, self._audio.shape[0] - frame_sample_num, frame_sample_num)):
                if t0 + length > self._audio.shape[0] - 1:
                    snippet = self._audio[t0:]
                    self._transformed[i, j] = np.abs(np.dot(wavelet[:snippet.shape[0]], snippet))
                else:
                    snippet = self._audio[t0:t0 + length]
                    self._transformed[i, j] = np.abs(np.dot(wavelet, snippet))
                self._progress.string(((i + 1)/self._transformed.shape[0] + j)/freqs.shape[0])
        self._progress.string(1)
