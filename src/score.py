from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
import copy
import logging
import numpy as np
import os
import traceback
from xml.etree.ElementTree import Element, ElementTree, parse as parse_etree

from .leftgoes import Progress


class Note:
    n256th: int = 1
    n128th: int = 2 * n256th
    n64th: int = 2 * n128th
    n32nd: int = 2 * n64th
    n16th: int = 2 * n32nd
    n8th: int = 2 * n16th
    n4th: int = 2 * n8th
    n2nd: int = 2 * n4th
    n1st: int = 2 * n2nd

    EIGHTH: int = n8th
    QUARTER: int = n4th
    HALF: int = n2nd
    WHOLE: int = n1st

    TRIPLET: float = 2/3

    durations: dict[str, int] = {'whole': n1st,
                                  'half': n2nd,
                               'quarter': n4th,
                                'eighth': n8th,
                                  '16th': n16th,
                                  '32nd': n32nd,
                                  '64th': n64th,
                                 '128th': n128th,
                                 '256th': n256th}


class Tremolo:
    def __init__(self) -> None:
        self.first: Element = None
        self.second: Element = None
        self.duration_start: float = None
        self.total_trem_duration: float = None
        self.trem_type: float = None
        self.on: bool = False
    
    def __bool__(self) -> bool:
        return self.on

    def __repr__(self) -> str:
        return f'Tremolo(duration_start={self.duration_start}, total_trem_duration={self.total_trem_duration}, on={self.on})'

    def new_first(self, first: Element, trem_type: float, duration_start: float, total_trem_duration: float) -> None:
        self.on = True
        self.first = first
        self.second = None
        self.duration_start = duration_start
        self.total_trem_duration = total_trem_duration
        self.trem_type = trem_type
    
    def new_second(self, second: Element) -> None:
        self.second = second

    def reset(self) -> None:
        self.on = False
        self.first = None
        self.second = None
        self.duration_start = None
        self.total_trem_duration = None
        self.trem_type = None

    def next(self, duration) -> bool:
        if not self.on: return False

        if duration - self.duration_start < self.total_trem_duration:
            return True
        else:
            self.reset()
            return False
    
    def get_visible_unvisible(self, duration) -> tuple[Element, Element] | tuple[None, None]:
        if duration - self.duration_start % (2 * self.trem_type) < self.trem_type:
            print('first, second')
            return self.first, self.second  # first should be visible, second is invisible
        else:
            print('second, first')
            return self.second, self.first


class Amusing:
    first_measure_num: int = 1
    musescore_executable_path: str = 'MuseScore3.exe'
    temp_filename: str = '.amusing_thread'
    tempdir = '__temp__'

    _input_types: set[str] = {'.mscx', '.mscz'}
    _ignore_in_measure: set[str] = {'stretch', 'startRepeat', 'endRepeat', 'MeasureNumber', 'LayoutBreak', 'vspacerUp', 'vspacerDown'}
    _chord_note_attrs: set[str] = {'Stem', 'Note/NoteDot', 'Note'}
    _grace_note: set[str] = {'grace4', 'acciaccatura', 'appoggiatura', 'grace8after', 'grace16', 'grace16after', 'grace32', 'grace32after'}

    def __init__(self, width: int, outdir: str = 'frames', *, 
                       threads: int = 8, log_file: str | None = None,
                       delete_temp: bool = True, print_progress: bool = True,
                       frame0: int = 0) -> None:
        self.width = width
        self.outdir = outdir
        self.threads = threads
        self.delete_temp = delete_temp
        self.print_progress = print_progress
        self.frame0 = frame0
        
        self.jobs: dict[int, int] = {}
        self._progress = Progress()

        self._basetree: ElementTree = None
        self._measures_num: int = None
        self._trees: list[ElementTree] = None
        self._timesigs: np.ndarray = None
        self._score_width: float = None
        self._page_num: int = None

        self._protected: list[set[Element]] = None

        self.__exceptions: list[list[Exception]] = [[] for _ in range(self.threads)]
        self.__filepath: str = None

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

    def _print_progress(self, *args, **kwargs) -> None:
        if self.print_progress: self._progress.string(*args, **kwargs)

    def _add_protected_chord(self, chord: Element) -> None:
        for tag in self._chord_note_attrs:
            self._protected.extend(chord.findall(tag))
    
    def _remove_protected_chord(self, chord: Element) -> None:
        if chord is None: return
        for tag in self._chord_note_attrs:
            for element in chord.findall(tag):
                self._protected.remove(element)

    def _set_visible(self, element: Element, n: int) -> None:
        if element in self._protected[n]:
            return
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
        root = self._basetree.getroot()
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

    def _thread(self, thread_index: int) -> int | None:
        try:
            temp_path = f'{self.tempdir}\\{self.temp_filename}-{thread_index:02d}.mscx'
            tree = self._trees[thread_index]
            root = tree.getroot()

            if thread_index == 0:
                self._convert(tree, self.frame0, 1, temp_path)
                self._progress.start()
                self._print_progress(0)

            frame: int = self.frame0 + 1
            page: int = 1
            newpage: bool = False
            tremolo = Tremolo()

            staves = [staff.findall('Measure') for staff in root.findall('Score/Staff')]
            for measure_index, measures in enumerate(zip(*staves)):  
                for measure in measures:
                    for voice in measure:
                        if voice.tag == 'LayoutBreak':
                            if voice.find('subtype').text == 'page':
                                newpage = True
                                break
                
                if measure_index in self.jobs:
                    subdivision = self.jobs[measure_index]
                    frame_count: int = round(self._timesigs[measure_index, 0] / subdivision)

                    for duration_index, max_duration in enumerate(np.linspace(0, self._timesigs[measure_index, 0], frame_count, endpoint=False)):
                        for staff_index, measure in enumerate(measures):
                            for voice in measure:
                                if voice.tag in self._ignore_in_measure: continue
                                elif voice.tag != 'voice': logging.warning(f'element with tag={voice.tag!r} in {measure_index=}, {staff_index=}')

                                tremolo.reset()
                                self._remove_protected_chord(tremolo.first)
                                self._remove_protected_chord(tremolo.second)

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
                                        if (duration_type := element.find('durationType').text) == 'measure': break

                                        if (dots := element.find('dots')) is None:
                                            dotted = 1
                                        else:
                                            dotted = sum(1/2**i for i in range(int(dots.text) + 1))
                                        chord_duration = tuplet * dotted \
                                                       * Note.durations[duration_type]
                                        
                                        print(tremolo.first, tremolo.second)
                                        if tremolo.first and not tremolo.second:
                                            tremolo.new_second(element)
                                            self._add_protected_chord(element)
                                        elif (subtype := element.find('Tremolo/subtype')) is not None and subtype.text[0] == 'c':
                                            tremolo.new_first(element, Note.WHOLE/int(subtype.text[1:]), duration, chord_duration)
                                            self._add_protected_chord(element)

                                        if tremolo:
                                            visible_chord, invisible_chord = tremolo.get_visible_unvisible(duration)
                                            print(frame, format(duration, '.1f'), visible_chord, invisible_chord)
                                            if visible_chord and invisible_chord:
                                                for chord_attr in self._chord_note_attrs:
                                                    if (chord_element := invisible_chord.find(chord_attr)) is not None:
                                                        if (existant_visible := chord_element.find('visible')) is None:
                                                            visible = Element('visible')
                                                            visible.text = '0'
                                                            chord_element.append(visible)
                                                        else:
                                                            existant_visible.text = '0'
                                                self._set_visible(visible_chord, thread_index)
                                            if not tremolo.next(duration):
                                                self._remove_protected_chord(tremolo.first)
                                                self._remove_protected_chord(tremolo.second)
                                                break
                                        
                                        duration += chord_duration

                                    for elem in self._generate_sublevels(element, 5):
                                        self._set_visible(elem, thread_index)

                                    if duration - 0.01 > max_duration:
                                        break
                        
                        if frame % self.threads == thread_index:
                            self._convert(tree, frame, page, temp_path)
                            if thread_index == 0:
                                self._print_progress((measure_index + duration_index / frame_count) / len(self.jobs), suffix='Thread 0')
                        frame += 1

                    for measure in measures:
                        for elem in self._generate_sublevels(measure, 7):
                            self._set_visible(elem, thread_index)
                else:
                    for measure in measures:
                        for elem in self._generate_sublevels(measure, 7):
                            self._set_visible(elem, thread_index)

                if newpage:
                    page += 1
                    newpage = False
        except KeyboardInterrupt:
            logging.error(f'KeyboardInterrupt in thread {thread_index}')
            return
        except Exception as e:
            self.__exceptions[thread_index].append(e)
            raise
        
        return frame

    def read_score(self, filepath: str) -> None:
        self.__filepath = os.path.splitext(filepath)
        if self.__filepath[1] not in self._input_types:
            return
        elif self.__filepath[1] == '.mscz':
            self.__convert(filepath, self.tempdir + '.score.mscx')
            self._basetree = parse_etree(self.tempdir + '.score.mscx')
        else:
            self._basetree = parse_etree(filepath)
        baseroot = self._basetree.getroot()

        self._score_width = float(baseroot.find('Score/Style/pageWidth').text)
        self._page_num = 1
        for element in self._generate_sublevels(baseroot, 8):
            if element.tag == 'LayoutBreak':
                if element.find('subtype').text == 'page':
                    self._page_num += 1

        self._measures_num = len(baseroot.find('Score/Staff').findall('Measure'))
        
        self._trees = [copy.deepcopy(self._basetree) for _ in range(self.threads)]
        self._protected = [set() for _ in range(self.threads)]

        for i, subtree in enumerate(self._trees):
            for element in self._generate_sublevels(subtree.getroot(), 8):
                if element.tag == 'Rest':
                    if element.find('visible') is not None: continue

                    self._protected[i].add(element)
                    visible = Element('visible')
                    visible.text = '0'
                    element.append(visible)
        self._read_timesigs()

    def add_job(self, measures: int | Sequence[int] | None, subdivision: float) -> None:
        if isinstance(measures, int):
            self.jobs.update({measures - self.first_measure_num: subdivision})
            logging.info('added job with 1 measure')
        else:
            if isinstance(measures, range):
                if measures.stop == -1:
                    measures = range(measures.start, self._measures_num + self.first_measure_num)
            self.jobs.update({measure - self.first_measure_num: subdivision for measure in measures})
            logging.info(f'added job with {len(measures)} measures')

    def add_job_all_measures(self, subdivision: float) -> None:
        self.jobs.update({i: subdivision for i in range(self._measures_num)})

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

        self._progress.start()
        self._sort_jobs()
        self._thread(0)
        # with ThreadPoolExecutor(self.threads) as executor:
        #     for i in range(self.threads):
        #         executor.submit(self._thread, i)
        
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

        self._print_progress(1, suffix='Main Thread')
