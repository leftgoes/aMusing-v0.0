import os
import logging
from numpy import arange
from xml.etree.ElementTree import Element, ElementTree

from leftgoes.utils import Progress, linmap
from mxml import MusicXML

class Note:
    WHOLE: int = 64
    HALF: int = 32
    QUARTER: int = 16
    EIGHTH: int = 8
    SIXTEENTH: int = 4
    THIRTYSECOND: int = 2
    SIXTYFOURTH: int = 1

    durations: dict[str, int] = {'whole': WHOLE,
                                 'half': HALF,
                                 'quarter': QUARTER,
                                 'eighth': EIGHTH,
                                 '16th': SIXTEENTH,
                                 '32nd': THIRTYSECOND,
                                 '64th': SIXTYFOURTH}

class Mscx(MusicXML):
    _temp_path = '__temp__.mscx'

    def __init__(self, subdivision_note: int, log_file: str | None = None) -> None:
        super().__init__(subdivision_note, log_file)
        self.subdivision_note = subdivision_note
        self.progress = None if log_file is None else Progress()

        self._tree: ElementTree

    @staticmethod
    def _set_visible(element: Element) -> None:
        for visible in element.findall('visible'):
            element.remove(visible)

    def _set_visible_spanner(self, spanner: Element) -> None:
        for element in spanner:
            self._set_visible(element)
            for elem in element:
                self._set_visible(elem)

    def _set_visible_measure(self, measure: Element) -> None:
        for voice in measure:
            for e1 in voice:
                for e2 in e1:
                    for e3 in e2:
                        for e4 in e3:
                            for e5 in e4:
                                self._set_visible(e5)
                            self._set_visible(e4)
                        self._set_visible(e3)
                    self._set_visible(e2)
                self._set_visible(e1)

    def _save(self, n: int, directory: str, dpi: int, single_page: bool, progress: float):
        self.write(self._temp_path, self._tree)
        to_file = os.path.join(directory, f'frm{n:04d}.png')
        self._export(self._temp_path, to_file, dpi=dpi)
        if single_page:
            if os.path.exists(to_file): os.remove(to_file)
            os.rename(os.path.join(directory, f'frm{n:04d}-1.png'), to_file)

        if self.progress is not None:
            self.progress.print(progress)

    def get_frames(self, directory: str = 'frames', dpi: int = 300, *, min_measure: int = 1, max_measure: int | None = None, n0: int = 1, single_page: bool = True, delete_temp: bool = True) -> int:
        root = self._tree.getroot()
        for score in root:
            if score.tag == 'Score': break
        else: logging.error('no score')
        
        if self.progress is not None: self.progress.start()
        n = n0
        
        staves_measures = [[measure for measure in staff if measure.tag == 'Measure'] for staff in score if staff.tag == 'Staff']
        if max_measure is None: max_measure = len(staves_measures[0])
        logging.info(f'{min_measure=}, {max_measure=}')
        measure_durations = [0 for _ in staves_measures]
        for m, measures_all_staves in enumerate(zip(*staves_measures)): 
            for staff_index, measure in enumerate(measures_all_staves):
                for not_chord in measure[0]:
                    if not_chord.tag == 'Chord': break
                    elif not_chord.tag == 'TimeSig':
                        measure_durations[staff_index] = Note.WHOLE * int(not_chord.find('sigN').text) // int(not_chord.find('sigD').text)
            
            for max_duration in arange(0, measure_durations[-1], self.subdivision_note):
                for measure_duration, measure in zip(measure_durations, measures_all_staves):
                    measure: Element
                    for voice in measure:
                        if voice.tag == 'voice':
                            duration, tuplet, dotted = 0, 1, 1
                            for element in voice:
                                if element.tag == 'location':
                                    duration += measure_durations[-1]/measure_duration * eval(element.find('fractions').text) * Note.WHOLE
                                elif element.tag == 'Tuplet':
                                    tuplet = int(element.find('normalNotes').text)/int(element.find('actualNotes').text)
                                    self._set_visible(element)
                                elif element.tag == 'endTuplet':
                                    tuplet = 1
                                elif element.tag == 'Chord' or element.tag == 'Rest':
                                    dots = element.find('dots')
                                    if dots is None: dotted = 1
                                    else: dotted = sum(1/2**i for i in range(int(dots.text) + 1))
                                    
                                    duration_type = element.find('durationType').text
                                    if element.tag == 'Chord':
                                        for elem in element:
                                            if elem.tag == 'Note':
                                                self._set_visible(elem)
                                                for e in elem:
                                                    if element.tag == 'Spanner':
                                                        self._set_visible_spanner(e)
                                                    else:
                                                        self._set_visible(e)
                                            elif element.tag == 'Spanner':
                                                self._set_visible_spanner(elem)
                                            else:
                                                self._set_visible(elem)
                                    else:
                                        self._set_visible(element)
                                        if dots is not None:
                                            self._set_visible(element.find('NoteDot'))
                                        if duration_type == 'measure': break
                                    duration += tuplet * dotted * measure_durations[-1]/measure_duration * Note.durations[duration_type]
                                elif element.tag == 'Spanner':
                                    self._set_visible_spanner(element)
                                else:
                                    logging.debug(f'{element.tag=}')
                                    self._set_visible(element)

                                if duration - 0.01 > max_duration:
                                    break

                        elif voice.tag == 'MeasureNumber':
                            self._set_visible(voice)
                        elif voice.tag == 'LayoutBreak':
                            continue
                        else:
                            logging.warning(f'{voice.tag=}')

                if m + 1 >= min_measure and m + 1 <= max_measure:
                    self._save(n, directory, dpi, single_page, linmap(m + 1 + max_duration/measure_durations[-1], (min_measure, max_measure + 1), (0, 1)))
                    n += 1

            for measure in measures_all_staves:
                self._set_visible_measure(measure)

        if delete_temp: self.remove_temp()
        return n
