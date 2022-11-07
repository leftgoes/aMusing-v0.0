from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
import copy
import logging
import numpy as np
import os
import time
from xml.etree.ElementTree import Element, ElementTree, parse as parse_etree


class Note:
    WHOLE: int = 128
    HALF: int = 64
    QUARTER: int = 32
    EIGHTH: int = 16
    SIXTEENTH: int = 8
    THIRTYSECOND: int = 4
    SIXTYFOURTH: int = 2
    HUNDREDTWENTYEIGTH: int = 1

    durations: dict[str, int] = {'whole': WHOLE,
                                 'half': HALF,
                                 'quarter': QUARTER,
                                 'eighth': EIGHTH,
                                 '16th': SIXTEENTH,
                                 '32nd': THIRTYSECOND,
                                 '64th': SIXTYFOURTH,
                                 '128th': HUNDREDTWENTYEIGTH}

class Amusing:
    temp_filename: str = '__temp__'
    musescore_executable_path: str = 'MuseScore3.exe'
    first_measure_num: int = 0
    ignore_in_measure: set[str] = {'startRepeat', 'endRepeat', 'LayoutBreak'}

    def __init__(self, width: int, threads: int = 8, log_file: str | None = None, delete_temp: bool = True) -> None:
        self.width = width
        self.threads = threads
        self.delete_temp = delete_temp
        self.jobs: dict[int, int] = {}

        self._tree: ElementTree = None
        self._timesigs: np.ndarray
        self._score_width: float

        logging.basicConfig(filename=log_file,
                            level=logging.WARNING,
                            format='[%(levelname)s:%(filename)s:%(lineno)d] %(message)s')

    @staticmethod
    def _set_visible(element: Element) -> None:
        for visible in element.findall('visible'):
            element.remove(visible)
    
    def _generate_sublevels(self, element: Element, n: int) -> Iterator[Element]:
        yield element
        if n >= 1:
            for elem in element:
                for e in self._generate_sublevels(elem, n - 1):
                    yield e

    def _remove_temp(self, filepath: str) -> None:
        if not self.delete_temp: return
        if os.path.exists(filepath):
            os.remove(filepath)
            logging.info(f'removed {filepath!r}')
        else:
            logging.warning(f'tried to remove {filepath!r} but not found')

    def _convert(self, tree: ElementTree, to_path: str, temp_path: str):
        self._write(tree, temp_path)
        self._export(temp_path, to_path)

    def _export(self, musicxml_path: str, to_path: str) -> None:
        os.system(f'{self.musescore_executable_path} {musicxml_path} --export-to {to_path} -r {self.width / self._score_width}')
        logging.info(f'exported {musicxml_path=!r} to {to_path=!r}')
    
    def _read_timesigs(self) -> None:
        root = self._tree.getroot()
        staves = root.findall('Score/Staff')
        self._timesigs = np.empty((len(staves[0].findall('Measure')), len(staves)))

        for j, staff in enumerate(staves):
            for i, measure in enumerate(staff.findall('Measure')):
                for element in measure[0]:
                    if element.tag == 'TimeSig':
                        timesig = Note.WHOLE \
                                * int(element.find('sigN').text) \
                                / int(element.find('sigD').text) 
                    elif element.tag == 'Chord': break
                self._timesigs[i, j] = timesig

    def _write(self, tree: ElementTree, filepath: str) -> None:
        tree.write(filepath, encoding='UTF-8', xml_declaration=True)
        logging.info(f'wrote tree to {filepath=!r}')

    def _thread(self, n: int) -> None:
        measure_indices = list(self.jobs)[n::self.threads]
        tree = copy.deepcopy(self._tree)
        root = tree.getroot()

        frame: int = 0

        staves = [staff.findall('Measure') for staff in root.findall('Score/Staff')]
        for measure_index, measures in enumerate(zip(*staves)):
            if measure_index not in self.jobs:
                continue

            frame_count: int = round(self._timesigs[measure_index, -1] / self.jobs[measure_index])
            if measure_index not in measure_indices:
                frame += frame_count
                for measure in measures:
                    for elem in self._generate_sublevels(measure, 7):
                        self._set_visible(elem)
                continue
            
            for max_duration in np.linspace(0, self._timesigs[measure_index, -1], frame_count, endpoint=False):
                logging.debug(f'{n=}, {measure_index=}, {staff_index=}, {frame_count=}, timesig={self._timesigs[measure_index, staff_index]}') 
                for staff_index, measure in enumerate(measures):
                    for voice in measure:
                        if voice.tag in self.ignore_in_measure: continue
                        elif voice.tag != 'voice': logging.warning(f'element with tag={voice.tag!r} in {measure_index=}, {staff_index=}')

                        duration, tuplet, dotted = 0, 1, 1
                        for element in voice:
                            if element.tag == 'location':
                                duration += eval(element.find('fractions').text) \
                                          * Note.WHOLE
                                    
                            elif element.tag == 'Tuplet':
                                tuplet = int(element.find('normalNotes').text)/int(element.find('actualNotes').text)
                            elif element.tag == 'endTuplet':
                                tuplet = 1
        
                            elif element.tag == 'Chord' or element.tag == 'Rest':
                                duration_type = element.find('durationType').text
                                if duration_type == 'measure': break

                                dots = element.find('dots')
                                if dots is None: dotted = 1
                                else: dotted = sum(1/2**i for i in range(int(dots.text) + 1))
                                
                                duration += tuplet * dotted \
                                          * Note.durations[duration_type]

                            for elem in self._generate_sublevels(element, 5):
                                self._set_visible(elem)

                            if duration - 0.01 > max_duration:
                                break
                
                self._convert(tree, f'C:\\Coding\\Music\\aMusing\\testing\\frm{frame:04d}.png', f'{self.temp_filename}_{n:02d}.mscx')
                frame += 1

            for measure in measures:
                for elem in self._generate_sublevels(measure, 7):
                    self._set_visible(elem)
            
        self._remove_temp(f'{self.temp_filename}_{n:02d}.mscx')

    def read(self, filepath: str) -> None:
        self._tree = parse_etree(filepath)
        self._score_width = float(self._tree.getroot().find('Score/Style/pageWidth').text)
        self._read_timesigs()

    def add_job(self, measures: int | Sequence[int], subdivision: int) -> None:
        if isinstance(measures, int):
            self.jobs.update({measures - self.first_measure_num: subdivision})
            logging.info('added job with 1 measure')
        else:
            self.jobs.update({measure - self.first_measure_num: subdivision for measure in measures})
            logging.info(f'added job with {len(measures)} measure')

    def generate_frames(self) -> None:
        logging.info(f'generate frames using {self.threads} process{"es" if self.threads != 1 else ""}')

        with ThreadPoolExecutor(self.threads) as executor:
            for i in range(self.threads):
                executor.submit(self._thread, i)
