# aMusing
- [Installation](#installation)
- [Example Code](#example-code)
    - [Score](#score)
    - [Munim](#munim)
- [Introduction](#introduction)
- [Animation](#animation)
    - [Score](#score-1)
    - [Mumin](#munim-1)
- [ToDo](#todo)
- [Inspiration](#inspiration)
- [Examples](#examples)

## Installation
```
pip install amusing
```

## Example Code

### Score

Get the individual frames of the score into output directory `outdir`:

```python
from amusing.score import Amusing, Note

WIDTH_IN_PIXELS: int = 1820
NUMBER_OF_THREADS: int = 8

MUSESCORE_FILEPATH: str = 'score.mscx'


amusing = Amusing(width=WIDTH_IN_PIXELS,
                  outdir='frames',
                  threads=NUMBER_OF_THREADS)
amusing.read_score(MUSESCORE_FILEPATH)
amusing.add_job(measures=1,
                subdivision=Note(16))
amusing.add_job(measures=[2, 3],
                subdivision=Note(4).triplet())
amusing.add_job(measures=range(4, 7),
                subdivision=Note(8).n_tuplet(5, 4))
amusing.generate_frames()
```

Delete all jobs:
```python
amusing.delete_jobs()
```

### Munim
*Mu*sic A*nim*ation

Render video of the frequency spectrum using [Morlet wavelet](https://en.wikipedia.org/wiki/Morlet_wavelet)
```python
from amusing.munim import Morlet

AUDIO_FILEPATH: str = 'example.mp3'
TO_VIDEO_FILEPATH: str = 'example.mp4'

morlet = Morlet(fps, width, height)
morlet.read_audio(AUDIO_FILEPATH)
morlet.transform()
morlet.render_video(TO_VIDEO_FILEPATH)
```

Using [Short-time Fourier Transform](https://en.wikipedia.org/wiki/Short-time_Fourier_transform) (STFT)
```python
from amusing.munim import STFT

morlet = STFT(fps, width, height)
morlet.read_audio(AUDIO_FILEPATH)
morlet.transform()
morlet.render_video(TO_VIDEO_FILEPATH)
``` 

[2d-Oscilloscope](https://en.wikipedia.org/wiki/Oscilloscope):
```python
from amusing.munim import Oscillate

oscillate = Oscillate(fps, width)
oscillate.read_audio(AUDIO_FILEPATH)
oscillate.render_video(TO_VIDEO_FILEPATH)
```

## Introduction
- programmatic animation of sheet music
    - notes appearing consecutively
    - uses [MuseScore](https://musescore.org/) as notation software

## Animation
### Score
- generates full resolution frames of the video
- **not** synchronized to audio

### Munim
- audio visualization
    - STFT
        - linear spectrum
    - Morlet (DWT)
        - arbitrary spectrum
        - implemented logarithmic
    - 2D Oscilloscope

## ToDo
- increase efficiency of making the score
    - OMR, e. g. [Audiveris](https://github.com/Audiveris)
- programmatically combine frames to high quality video
    - automatic synchronization to audio
        - DWT where Wavelet function is a sample of the instrument sound
        - Deconvolution of overtones
- add background interest

## Inspiration
[![Chopin Prelude 16 ANIMATED](https://img.youtube.com/vi/kq6BofwPSJI/maxresdefault.jpg)](https://www.youtube.com/kq6BofwPSJI)

## Examples
[![Chopin op. 25 no. 11](https://img.youtube.com/vi/9X8dbjO-wt4/maxresdefault.jpg)](https://youtu.be/9X8dbjO-wt4)
