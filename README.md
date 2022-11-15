# aMusing
- [Installation](#installation)
- [Introduction](#introduction)
- [Animation](#animation)
    - [Amusing](#amusing-1)
    - [Mumin](#munim)
- [ToDo](#todo)
- [Inspiration](#inspiration)
- [Examples](#examples)

## Installation
```python
pip install amusing
```

## Introduction
- programmatic animation of sheet music
  - notes appearing consecutively
  - uses [MuseScore](https://musescore.org/) as notation software

## Animation
### Amusing
- generates full resolution frames of the video
- **not** synchronized to audio

### Munim
- audio visualization
    - STFT
        - linear spectrum
    - Morlet (DWT)
        - arbitrary spectrum
        - implemented logarithmic

## ToDo
- increase efficiency of making the score
    - OMR, e. g. [Audiveris](https://github.com/Audiveris)
- programmatically combine frames to high quality video
    - automatic synchronization to audio
        - DWT where Wavelet function is based on piano sound
        - Deconvolution of overtones
- add background interest

## Inspiration
[![Chopin Prelude 16 ANIMATED](https://img.youtube.com/vi/kq6BofwPSJI/maxresdefault.jpg)](https://www.youtube.com/kq6BofwPSJI)

## Examples
[![Chopin op. 25 no. 11](https://img.youtube.com/vi/9X8dbjO-wt4/maxresdefault.jpg)](https://youtu.be/9X8dbjO-wt4)
