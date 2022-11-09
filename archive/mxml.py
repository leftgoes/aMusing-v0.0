import copy
import logging
import os
from time import sleep
from typing import Callable, Literal

import xml.etree.ElementTree as ETree
from xml.etree.ElementTree import Element, ElementTree


class MusicXML:
    _removing_direction: set = {'bracket', 'dashes', 'dynamics', 'words', 'octave-shift', 'pedal', 'wedge'}
    _removing_note: list = ['accidental', 'beam', 'notations']
    _temp_path = '__temp__.musicxml'

    def __init__(self, subdivision_note: float, log_file: str | None = None) -> None:
        logging.basicConfig(filename=log_file, level=logging.INFO, format='[%(levelname)s:%(filename)s:%(lineno)d] %(message)s')
        self.subdivision_note = subdivision_note

        self.crochet: int = None
        self._tree: ElementTree
        self._processed_tree: ElementTree

    @staticmethod
    def altering(key: int) -> Callable[[Literal['C', 'D', 'E', 'F', 'G', 'A', 'B']], Literal['-1', '0', '1']]:
        if key == 0: return lambda _: 0
        sign = key//abs(key)
        if abs(key) > 7: logging.warning(f'key signature with {abs(key)} {"sharps" if sign == 1 else "flats"}')
        
        alter = [0 for _ in range(7)]
        i = -1 if sign == 1 else 3
        for _ in range(abs(key)):
            i = (i + sign * 4) % 7
            alter[i] += sign
        alter = [str(i) for i in alter]
        
        return lambda step: alter['CDEFGAB'.index(step)]

    @property
    def subdivision(self) -> int:
        return round(4 * self.crochet * self.subdivision_note)

    def remove_temp(self) -> None:
        if os.path.exists(self._temp_path):
            os.remove(self._temp_path)
            logging.info(f'removed {self._temp_path!r}')
        else:
            logging.warning(f'tried to remove {self._temp_path!r} but not found')

    def _get_crochet_duration(self) -> None:
        root = self._tree.getroot()
        for part in root:
            if part.tag != 'part': continue
            for measure in part:
                for note in measure:
                    if note.tag != 'note': continue
                    subdivision_note = note.find('type')
                    if subdivision_note is None: continue
                    if subdivision_note.text != 'quarter': continue
                    if note.find('time-modification') is not None: continue
                    self.crochet = int(note.find('duration').text)
                    logging.info(f'{self.crochet=}')
                    return

    def _keep_note_max_measure(self, max_subdivision: int, duration: int, duration_elem: Element) -> bool:
        duration_offset = 0 if duration_elem is None else int(duration_elem.text)
        return max_subdivision == -1 or duration - duration_offset <= max_subdivision * self.subdivision

    def _export(self, from_file: str, to_file: str, dpi: int = 300) -> None:
        os.system(f'MuseScore3.exe {from_file} --export-to {to_file} -r {dpi}')
        logging.info(f'exported {from_file=!r} to {to_file=!r} with {dpi=}')

    def _process(self, max_measure: int, max_subdivision: int = -1, *, root: Element | None = None) -> None:
        if root is None:
            root = copy.deepcopy(self._tree.getroot())
        for branch in root:
            if branch.tag == 'part':
                part = branch
                logging.info(f'found part with id={part.attrib["id"]!r}')
                break
        else:
            logging.error(f'could not find \'part\' in root')
            return

        for measure in part:
            n = int(measure.attrib['number'])
            if measure.tag != 'measure':
                logging.warning(f'found non-measure in {part=}')

            for attributes in measure.findall('attributes'):
                if attributes is not None:
                    key = attributes.find('key')
                    if key is not None:
                        fifths = key.find('fifths')
                        if fifths is not None:
                            logging.debug(f'measure {n}, new_key={fifths.text}')
                            if n <= max_measure:
                                altering = self.altering(int(fifths.text))
                            else:
                                attributes.remove(key)
                    
                    if n > max_measure:
                        for clef in attributes.findall('clef'): attributes.remove(clef)

                        time_elem = measure.find('attributes/time')
                        if time_elem is not None:
                            time_elem.attrib.update({'print-object': 'no'})
                    if len(attributes) == 0: remove_measure.append(attributes)

            if n < max_measure: continue  # keep in score

            remove_measure: list[Element] = []
            for direction in measure:
                if direction.tag != 'direction': continue
                remove_measure.append(direction)

            duration, time_mod = 0, 0
            inside_tuplet: bool = False
            keep_first_tuplet: bool = False
            for note in measure:
                if note.find('chord') is None:
                    duration_elem = note.find('duration')
                    if note.tag == 'backup':
                        duration -= int(duration_elem.text)
                    else:
                        duration_elem = note.find('duration')
                        if duration_elem is not None:
                            duration += int(duration_elem.text)
                
                if note.tag != 'note': continue
                time_mod_elem = note.find('time-modification')

                if time_mod_elem is not None:  # note is tuplet
                    if n == max_measure:
                        keep_note = self._keep_note_max_measure(max_subdivision, duration, duration_elem)
                        if not inside_tuplet: keep_first_tuplet = keep_note
                        inside_tuplet = True

                        if keep_note: continue
                    
                    # converts tuplets to normal notes and delete if too many
                    if n != max_measure or (n == max_measure and not keep_first_tuplet):
                        normal = int(time_mod_elem.find('normal-notes').text)
                        actual = int(time_mod_elem.find('actual-notes').text)
                        time_mod += 1 - normal/actual
                        if time_mod >= 1:
                            remove_measure.append(note)
                            time_mod -= 1
                        else:
                            duration_elem.text = str(round(int(duration_elem.text) * actual / normal))
                            note.remove(time_mod_elem)
                else:
                    if time_mod != 0:
                        logging.warning(f'time_mod should be zero but has value {time_mod}')
                    inside_tuplet = False

                if n == max_measure and self._keep_note_max_measure(max_subdivision, duration, duration_elem):
                    continue

                note.attrib.update({'print-object': 'no'})

                for t in self._removing_note:
                    elems = note.findall(t)
                    for e in elems: note.remove(e)

                if note.find('rest') is not None: continue

                stem = note.find('stem')
                if stem is not None: stem.text = 'none'

                step_text = note.find('pitch/step').text
                alter = note.find('pitch/alter')
                new_alter_value = altering(step_text)
                if alter is None:
                    if new_alter_value != '0':
                        new_alter = Element('alter')
                        new_alter.text = altering(step_text)
                        note.find('pitch').insert(1, new_alter)
                else: 
                    if new_alter_value == alter.text:
                        alter.text = new_alter_value
                    else:
                        alter.text = new_alter_value
            
            for direction_or_note in remove_measure:
                measure.remove(direction_or_note)
            
        self._processed_tree = ElementTree(root)
        logging.info(f'parsed to tree')

    def write(self, file: str, tree: ElementTree) -> None:
        tree.write(file, encoding='UTF-8', xml_declaration=True)
        logging.info(f'wrote tree to {file=!r}')
    
    def convert(self, to_file: str, max_measure: int, max_subdivision: int = -1, dpi: int = 300, delete_temp: bool = True, *, root: Element | None = None, single_page: bool = False) -> None:
        self._process(max_measure, max_subdivision, root=root)
        self.write(self._temp_path, self._processed_tree)
        self._export(self._temp_path, to_file, dpi=dpi)

        if single_page:
            to_file_split = os.path.splitext(to_file)
            from_file = f'{to_file_split[0]}-1{to_file_split[1]}'
            if os.path.exists(to_file):
                if not os.path.exists(from_file):
                    logging.error(f'did not find \'{to_file_split[0]}-1{to_file_split[1]}, MuseScore3 conversion probably failed, for more infos import {self._temp_path!r} in Musescore')
                    return
                logging.info(f'overriding {to_file!r}')
                os.remove(to_file)
            os.rename(f'{to_file_split[0]}-1{to_file_split[1]}', to_file)
                
        if delete_temp: self.remove_temp()

    def read(self, file: str) -> None:
        self._tree = ETree.parse(file)
        self._get_crochet_duration()

    def get_frames(self, directory: str = 'frames', dpi: int = 300, single_page: bool = False, *, out_names_by: Literal['measure', 'index'] = 'index', first_measure: int = 1) -> None:
        root = self._tree.getroot()
        for branch in root:
            if branch.tag == 'part':
                part = branch
                break
        else:
            logging.error('not part in root')
            return

        subdivisions: list[int] = []
        for measure in part:
            time_elem = measure.find('attributes/time')
            if time_elem is not None:
                p, q = tuple(int(time_elem.find(i).text) for i in ('beats', 'beat-type'))
            subdivisions.append(round(p/q * 4*self.crochet / self.subdivision))

        if not os.path.exists(directory):
            logging.info(f'created new directory {os.path.join(os.getcwd(), directory)}')
            os.mkdir(directory)

        i = 0
        for measure, sub_num in zip(part[first_measure - 1:], subdivisions[first_measure - 1:]):
            n = int(measure.attrib['number'])
            for sub in range(sub_num):
                logging.info(f'creating frame with max_measure={n}, max_subdivision={sub} out of a total of {len(part)=} measures')
                fname = f'm{n:03d}s{sub:03d}.png' if out_names_by == 'measure' else f'frm{i:04d}.png'
                self.convert(f'{directory}\\{fname}', n, sub, dpi=dpi, delete_temp=False, root=copy.deepcopy(root), single_page=single_page)
                i += 1
        
        self.remove_temp()
 
