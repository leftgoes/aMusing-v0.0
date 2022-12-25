from typing import Literal as L

StrPath = str
IntYPosition = int
StrKType = L['position', 'time']

first_measure_num: int = 1

def frame_index_to_path(index: int, page: int | None = None) -> StrPath:
    if page is None:
        return f'frm{index:04d}.png'
    else:
        return f'frm{index:04d}-{page}.png'

def frame_path_to_index(filepath: StrPath) -> int:
    return int(filepath[3:-4])