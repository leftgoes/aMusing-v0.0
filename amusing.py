from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
import copy
import logging
import numpy as np
import os
import traceback
from xml.etree.ElementTree import Element, ElementTree, parse as parse_etree

from leftgoes.utils import Progress


class Note:
    n256: int = 1
    n128: int = 2 * n256
    n64: int = 2 * n128
    n32: int = 2 * n64
    n16: int = 2 * n32
    EIGHTH: int = 2 * n16
    QUARTER: int = 2 * EIGHTH
    HALF: int = 2 * QUARTER
    WHOLE: int = 2 * HALF

    durations: dict[str, int] = {'whole': WHOLE,
                                  'half': HALF,
                               'quarter': QUARTER,
                                'eighth': EIGHTH,
                                  '16th': n16,
                                  '32nd': n32,
                                  '64th': n64,
                                 '128th': n128,
                                 '256th': n256}

class Amusing:
    first_measure_num: int = 0
    musescore_executable_path: str = 'MuseScore3.exe'
    temp_filename: str = '.amusing_thread'
    tempdir = '__temp__'

    _input_types: set[str] = {'.mscx', '.mscz'}
    _ignore_in_measure: set[str] = {'stretch', 'startRepeat', 'endRepeat', 'LayoutBreak', 'vspacerUp', 'vspacerDown'}
    _grace_note: set[str] = {'grace4', 'acciaccatura', 'appoggiatura', 'grace8after', 'grace16', 'grace16after', 'grace32', 'grace32after'}

    def __init__(self, width: int, outdir: str = 'frames', *, 
                       threads: int = 8, log_file: str | None = None,
                       delete_temp: bool = True) -> None:
        self.width = width
        self.outdir = outdir
        self.threads = threads
        self.delete_temp = delete_temp
        
        self.jobs: dict[int, int] = {}
        self.progress = Progress()

        self._tree: ElementTree = None
        self._timesigs: np.ndarray = None
        self._score_width: float = None
        self._page_num: int = None
        self._protected: set[Element] = None

        self.__exceptions: list[list[Exception]] = [[] for _ in range(self.threads)]
        self.__filepath: str = None
        self.__print_thread: int = 0

        logging.basicConfig(filename=log_file,
                            level=logging.WARNING,
                            format='[%(levelname)s:%(filename)s:%(lineno)d] %(message)s')

    @staticmethod
    def __convert(from_path: str, to_path: str, dpi: int | None = None) -> None:
        cmd = f'{Amusing.musescore_executable_path} {from_path} --export-to {to_path}'
        if dpi is not None:
            cmd += f' -r {dpi}'
        os.system(cmd)

    @staticmethod
    def _remove_temp(filepath: str) -> bool:
        if os.path.exists(filepath):
            os.remove(filepath)
            logging.info(f'removed {filepath!r}')
            return True
        else:
            logging.warning(f'tried to remove {filepath!r} but not found')
            return False
    
    @staticmethod
    def _set_invisible(element: Element) -> None:
        visible = Element('visible')
        visible.text = '0'
        element.insert(0, visible)

    def _set_visible(self, element: Element) -> None:
        if element in self._protected: return
        for visible in element.findall('visible'):
            element.remove(visible)

    def _generate_sublevels(self, element: Element, n: int) -> Iterator[Element]:
        yield element
        if n >= 1:
            for elem in element:
                for e in self._generate_sublevels(elem, n - 1):
                    yield e
    
    def _sort_jobs(self) -> None:
        self.jobs = dict(sorted(self.jobs.items()))

    def _convert(self, tree: ElementTree, frame: int, page: int, temp_path: str) -> None:
        self._write(tree, temp_path)

        to_file = os.path.join(self.outdir, f'frm{frame:04d}.png') 
        self._export(temp_path, to_file)
        for i in range(1, self._page_num + 1):
            if i == page:
                if os.path.exists(to_file): os.remove(to_file)
                os.rename(os.path.join(self.outdir, f'frm{frame:04d}-{page}.png'), to_file)
            else:
                os.remove(os.path.join(self.outdir, f'frm{frame:04d}-{i}.png'))

    def _export(self, musicxml_path: str, to_path: str) -> None:
        self.__convert(musicxml_path, to_path, self.width / self._score_width)
        logging.info(f'exported {musicxml_path=!r} to {to_path=!r}')
    
    def _read_timesigs(self) -> None:
        root = self._tree.getroot()
        staves = root.findall('Score/Staff')
        self._timesigs = np.empty((len(staves[0].findall('Measure')), len(staves)))

        for j, staff in enumerate(staves):
            for i, measure in enumerate(staff.findall('Measure')):
                for element in measure.find('voice'):
                    if element.tag == 'TimeSig':
                        timesig = Note.WHOLE \
                                * int(element.find('sigN').text) \
                                / int(element.find('sigD').text) 
                    elif element.tag == 'Chord': break
                self._timesigs[i, j] = timesig
                if 'len' in measure.attrib:
                    self._timesigs[i, j] = Note.WHOLE * eval(measure.attrib['len'])

    def _write(self, tree: ElementTree, filepath: str) -> None:
        tree.write(filepath, encoding='UTF-8', xml_declaration=True)
        logging.info(f'wrote tree to {filepath=!r}')

    def _thread(self, n: int) -> int | None:
        try:
            temp_path = f'{self.tempdir}\\{self.temp_filename}-{n:02d}.mscx'
            measure_indices = list(self.jobs)[n::self.threads]
            tree = copy.deepcopy(self._tree)
            root = tree.getroot()

            frame: int = 0
            page: int = 1
            newpage: bool = False

            staves = [staff.findall('Measure') for staff in root.findall('Score/Staff')]
            for measure_index, measures in enumerate(zip(*staves)):  
                for measure in measures:
                    for voice in measure:
                        if voice.tag == 'LayoutBreak':
                            if voice.find('subtype').text == 'page':
                                newpage = True
                                break
                
                if measure_index in self.jobs:
                    frame_count: int = round(self._timesigs[measure_index, 0] / self.jobs[measure_index])
                    if measure_index not in measure_indices:
                        frame += frame_count
                        for measure in measures:
                            for elem in self._generate_sublevels(measure, 7):
                                self._set_visible(elem)

                    else:
                        for duration_index, max_duration in enumerate(np.linspace(0, self._timesigs[measure_index, 0], frame_count, endpoint=False)):
                            for staff_index, measure in enumerate(measures):
                                for voice in measure:
                                    if voice.tag in self._ignore_in_measure: continue
                                    elif voice.tag != 'voice': logging.warning(f'element with tag={voice.tag!r} in {measure_index=}, {staff_index=}')

                                    duration, tuplet, dotted = 0, 1, 1
                                    for element in voice:
                                        if any(element.find(tag) is not None for tag in self._grace_note):
                                            pass

                                        elif element.tag == 'location':
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
                            
                            if n == self.__print_thread:
                                self.progress.string((measure_indices.index(measure_index) + duration_index / frame_count) / len(measure_indices), suffix=f'Thread {n}')
                                self.__print_thread = 0 if n >= self.threads - 1 else n + 1
                            self._convert(tree, frame, page, temp_path)
                            frame += 1

                        for measure in measures:
                            for elem in self._generate_sublevels(measure, 7):
                                self._set_visible(elem)

                if newpage:
                    page += 1
                    newpage = False
        except KeyboardInterrupt:
            logging.error(f'KeyboardInterrupt in thread {n}')
            return
        except Exception as e:
            self.__exceptions[n].append(e)
            return
        
        return frame

    def read(self, filepath: str) -> None:
        self.__filepath = os.path.splitext(filepath)
        if self.__filepath[1] not in self._input_types:
            return
        elif self.__filepath[1] == '.mscz':
            self.__convert(filepath, self.tempdir + '.score.mscx')
            self._tree = parse_etree(self.tempdir + '.score.mscx')
        else:
            self._tree = parse_etree(filepath)
        root = self._tree.getroot()

        self._score_width = float(root.find('Score/Style/pageWidth').text)
        self._page_num = 1
        self._protected = set()
        for element in self._generate_sublevels(root, 8):
            if element.tag == 'LayoutBreak':
                if element.find('subtype').text == 'page':
                    self._page_num += 1
            elif element.tag == 'Rest':
                if element.find('visible') is None:
                    self._protected.add(element)
                    self._set_invisible(element)
        print(len(self._protected))
        self._read_timesigs()

    def add_job(self, measures: int | Sequence[int] | None, subdivision: float) -> None:
        if isinstance(measures, int):
            self.jobs.update({measures - self.first_measure_num: subdivision})
            logging.info('added job with 1 measure')
        else:
            self.jobs.update({measure - self.first_measure_num: subdivision for measure in measures})
            logging.info(f'added job with {len(measures)} measures')

    def delete_jobs(self) -> None:
        self.jobs = {}
    
    def generate_frames(self) -> None:
        logging.info(f'generate frames using {self.threads} process{"es" if self.threads != 1 else ""}')
    
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)
            logging.info(f'create outdir={self.outdir}')

        if not os.path.exists(self.tempdir):
            os.mkdir(self.tempdir)
            logging.info(f'create tempdir={self.tempdir}')

        self.progress.start()
        self._sort_jobs()
        with ThreadPoolExecutor(self.threads) as executor:
            results = [executor.submit(self._thread, i) for i in range(self.threads)]
        
        if sum(len(thread) for thread in self.__exceptions) > 0:
            for n, thread in enumerate(self.__exceptions):
                if len(thread) == 0: continue
                print('\n' + f'Exceptions Thread {n}'.center(60, '.'))
                for e in thread:
                    traceback.print_tb(e.__traceback__)
            quit()

        if self.delete_temp:
            logging.info('remove temp files')
            for n in range(self.threads):
                self._remove_temp(f'{self.tempdir}\\{self.temp_filename}-{n:02d}.mscx')
            if self.__filepath[1] == '.mscz':
                self._remove_temp(self.tempdir + '.score.mscx')
        
        if len(os.listdir(self.tempdir)) == 0:
            os.rmdir(self.tempdir)
            logging.info('remove tempdir')

        self.progress.string(1, suffix='Main Thread')
