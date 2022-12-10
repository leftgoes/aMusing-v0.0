from collections.abc import Iterator, Sequence
import copy
import logging
import numpy as np
import os
from time import sleep
from threading import Thread
from xml.etree.ElementTree import ElementTree

from .leftgoes import Progress
from .mscx import MElement, parse_custom_etree, Note


class Amusing:
    first_measure_num: int = 1
    musescore_executable_path: str = 'MuseScore3.exe'
    temp_filename: str = '.amusing_thread'
    tempdir = '__temp__'

    def __init__(self, width: int, outdir: str = 'frames', *, 
                       threads: int = 8, log_file: str | None = None,
                       delete_temp: bool = True, print_progress: bool = True,
                       first_emtpy_frame: bool = True, frame0: int = 0) -> None:
        self.width = width
        self.outdir = outdir
        self.threads = threads
        self.delete_temp = delete_temp
        self.print_progress = print_progress
        self.first_emtpy_frame = first_emtpy_frame
        self.frame0 = frame0
        
        self.jobs: dict[int, Note] = {}
        self._progress = Progress()
        self._tree: ElementTree = None
        
        self._measures_num: int = None
        self._score_width: float = None
        self._timesigs: np.ndarray = None
        self._page_num: int = None
        self._protected_tremolos: set[MElement] = set()

        self._filepath: tuple[str, str] = None

        logging.basicConfig(filename=log_file,
                            level=logging.WARNING,
                            format='[%(levelname)s:%(filename)s:%(lineno)d] %(message)s')

    @classmethod
    def convert(cls, from_path: str, to_path: str, dpi: int | None = None) -> None:
        cmd = f'{cls.musescore_executable_path} {from_path} --export-to {to_path}'
        if dpi is not None:
            cmd += f' -r {dpi}'
        os.system(cmd)

    @staticmethod
    def _last_element_in_measure(duration: float, max_duration) -> bool:
        return duration - 0.01 > max_duration

    @staticmethod
    def remove_file(filepath: str) -> bool:
        if os.path.exists(filepath):
            os.remove(filepath)
            logging.info(f'removed {filepath!r}')
            return True
        else:
            logging.warning(f'tried to remove {filepath!r} but not found')
            return False

    def temp_path(self, thread_index: int) -> str:
        return f'{self.tempdir}\\{self.temp_filename}_Thread-{thread_index:02d}.mscx'

    def _print_progress(self, *args, **kwargs) -> None:
        if self.print_progress: self._progress.string(*args, **kwargs)

    def _add_protected_chord(self, chord: MElement) -> None:
        for element in chord.get_chord_subelements():
            self._protected_tremolos.add(element)
    
    def _sort_jobs(self) -> None:
        self.jobs = dict(sorted(self.jobs.items()))

    def _convert(self, index: int, frame: int, page: int, tree: ElementTree) -> None:
        temp_path = self.temp_path(index)
        self._write(tree, temp_path)

        to_file = os.path.join(self.outdir, f'frm{frame:04d}.png') 
        self._export(temp_path, to_file)
        for i in range(1, self._page_num + 1):
            if i == page:
                if os.path.exists(to_file): os.remove(to_file)
                os.rename(os.path.join(self.outdir, f'frm{frame:04d}-{page}.png'), to_file)
            else:
                os.remove(os.path.join(self.outdir, f'frm{frame:04d}-{i}.png'))

    def _export(self, from_musescore_path: str, to_path: str) -> None:
        type(self).convert(from_musescore_path, to_path, self.width / self._score_width)
        logging.info(f'exported {from_musescore_path=!r} to {to_path=!r}')
    
    def _read_timesigs(self) -> None:
        root = self._tree.getroot()
        staves = root.findall('Score/Staff')
        self._timesigs = np.empty((len(staves[0].findall('Measure')), len(staves)))

        for j, staff in enumerate(staves):
            for i, measure in enumerate(staff.findall('Measure')):
                for element in measure.find('voice'):
                    if element.tag == 'TimeSig':
                        timesig = Note(1).value \
                                * int(element.find('sigN').text) \
                                / int(element.find('sigD').text) 
                    elif element.tag == 'Chord': break
                self._timesigs[i, j] = timesig
                if 'len' in measure.attrib:
                    self._timesigs[i, j] = Note(1).value * eval(measure.attrib['len'])

    def _write(self, tree: ElementTree, to_temp_path: str) -> None:
        tree.write(to_temp_path, encoding='UTF-8', xml_declaration=True)
        logging.info(f'wrote tree to {to_temp_path=!r}')

    def _get_trees(self, max_tremolo: Note) -> Iterator[tuple[int, ElementTree]]:
        root = self._tree.getroot()

        page: int = 1
        newpage: bool = False

        self._progress.start()
        self._print_progress(0)

        if self.first_emtpy_frame:
            yield 1, copy.deepcopy(self._tree)
            
        staves = [staff.findall('Measure') for staff in root.findall('Score/Staff')]
        for measure_index, measures in enumerate(zip(*staves)):  
            measures: tuple[MElement]
            for measure in measures:
                for voice in measure:
                    if voice.tag == 'LayoutBreak':
                        if voice.find('subtype').text == 'page':
                            newpage = True
                            break
            
            if measure_index in self.jobs:
                subdivision = self.jobs[measure_index]
                time_sig = self._timesigs[measure_index, 0]
                frame_count: int = round(time_sig / subdivision.value)

                for duration_index, max_duration in enumerate(np.linspace(0, time_sig, frame_count, endpoint=False)):
                    self._protected_tremolos.clear()
                    for staff_index, measure in enumerate(measures):
                        for voice in measure:
                            voice: MElement
                            if voice.is_unprintable(): continue
                            elif voice.tag != 'voice': logging.warning(f'element with tag={voice.tag!r} in {measure_index=}, {staff_index=}')

                            foundtremolo: bool = False
                            duration, tuplet, dotted = 0, 1, 1
                            for element_index, element in enumerate(voice):
                                element: MElement
                                if foundtremolo:
                                    foundtremolo = False
                                    continue

                                if element.is_gracenote():
                                    pass

                                elif element.tag == 'location':
                                    duration += eval(element.find('fractions').text) * Note(1).value
                                        
                                elif element.tag == 'Tuplet':
                                    tuplet = element.tuplet_value()
                                
                                elif element.tag == 'endTuplet':
                                    tuplet = 1

                                elif element.tag == 'Chord' or element.tag == 'Rest':
                                    dotted, duration_type = element.duration_value(time_sig)
                                    chord_duration = tuplet * dotted * duration_type

                                    tremolo = element.find('Tremolo')
                                    if tremolo is not None and (tremolo_subtype := tremolo.find('subtype').text)[0] == 'c':
                                        next_element: MElement = voice[element_index + 1]

                                        if max_duration < chord_duration + duration and (tremolo_note := int(tremolo_subtype[1:])) <= max_tremolo.value:
                                            tremolo_timediff = Note(1).value/tremolo_note * min(1, duration_type/Note(4).value)
                                            if ((max_duration - duration) % (2 * tremolo_timediff)) / tremolo_timediff < 1:
                                                element.set_visible_chord()
                                                next_element.set_invisible_chord()
                                            else:
                                                element.set_invisible_chord()
                                                next_element.set_visible_chord()

                                            for parent in (element, next_element):
                                                for subelement in parent.get_chord_subelements():
                                                    self._protected_tremolos.add(subelement)
                                            
                                            foundtremolo = True
                                            duration += chord_duration
                                    else:
                                        self._protected_tremolos.clear()
                                        duration += chord_duration   
                                
                                element.set_visible_all()

                                if self._last_element_in_measure(duration, max_duration):
                                    break
                    
                    yield page, copy.deepcopy(self._tree)
                    self._print_progress((list(self.jobs.keys()).index(measure_index) + duration_index / frame_count) / len(self.jobs))

                for measure in measures:
                    measure.set_visible_all()
            else:
                for measure in measures:
                    measure.set_visible_all()

            if newpage:
                page += 1
                yield page, copy.deepcopy(self._tree)
                newpage = False

    def read_score(self, filepath: str) -> None:
        self._filepath = os.path.splitext(filepath)
        if self._filepath[1] not in ('.mscx', '.mscz'):
            return
        elif self._filepath[1] == '.mscz':
            type(self).convert(filepath, self.tempdir + '.score.mscx')
            self._tree = parse_custom_etree(self.tempdir + '.score.mscx')
        else:
            self._tree = parse_custom_etree(filepath)
        baseroot = self._tree.getroot()

        self._score_width = float(baseroot.find('Score/Style/pageWidth').text)
        self._page_num = 1
        for element in baseroot.iter():
            if element.tag == 'LayoutBreak':
                if element.find('subtype').text == 'page':
                    self._page_num += 1

        self._measures_num = len(baseroot.find('Score/Staff').findall('Measure'))
        
        self._protected_tremolos = [set() for _ in range(self.threads)]

        for element in baseroot.iter():
            element: MElement
            if element.is_invisiblity_allowed():
                if not element.is_visible(): continue

                element.set_invisible()
                element.protect()
        self._read_timesigs()

    def add_job(self, measures: int | Sequence[int] | None, subdivision: Note) -> None:
        if isinstance(measures, int):
            self.jobs.update({measures - self.first_measure_num: subdivision})
            logging.info('added job with 1 measure')
        else:
            if isinstance(measures, range):
                if measures.stop == -1:
                    measures = range(measures.start, self._measures_num + self.first_measure_num)
            self.jobs.update({measure - self.first_measure_num: subdivision for measure in measures})
            logging.info(f'added job with {len(measures)} measures')

    def add_job_all_measures(self, subdivision: Note) -> None:
        self.jobs.update({i: subdivision for i in range(self._measures_num)})

    def delete_jobs(self) -> None:
        self.jobs = {}
    
    def generate_frames(self, max_tremolo: Note | None = None) -> None:
        if max_tremolo is None: max_tremolo = Note(16)
        logging.info(f'generate frames using {self.threads} process{"es" if self.threads != 1 else ""}')
    
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)
            logging.info(f'create outdir={self.outdir}')

        if not os.path.exists(self.tempdir):
            os.mkdir(self.tempdir)
            logging.info(f'create tempdir={self.tempdir}')

        self._progress.start()
        self._sort_jobs()
        threads: list[Thread] = [Thread() for _ in range(self.threads)]
        for frame, (page, tree) in enumerate(self._get_trees(max_tremolo), start=self.frame0):
            while (free_thread := next((t for t in threads if not t.is_alive()), None)) is None:
                sleep(0.01)
            thread_index = threads.index(free_thread)
            
            threads[thread_index] = Thread(target=self._convert, name=f'Thread {thread_index}', args=(thread_index, frame, page, tree))
            threads[thread_index].start()
        
        for thread in threads:
            thread.join()

        if self.delete_temp:
            logging.info('remove temp files')
            for thread_index in range(self.threads):
                self.remove_file(self.temp_path(thread_index))
            if self._filepath[1] == '.mscz':
                self.remove_file(self.tempdir + '.score.mscx')
        
        if len(os.listdir(self.tempdir)) == 0:
            os.rmdir(self.tempdir)
            logging.info('remove tempdir')

        self._print_progress(1)
