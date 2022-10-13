import copy
import logging
import os
from typing import Callable, Literal

import xml.etree.ElementTree as ETree
from xml.etree.ElementTree import Element, ElementTree


class MuseScore:
    _removing_direction: set = {'dynamics', 'words', 'octave-shift', 'pedal', 'wedge'}
    _removing_note: list = ['accidental', 'beam', 'notations']
    _temp_path = '__temp__.musicxml'

    def __init__(self, note_type: float, log_file: str | None = None) -> None:
        logging.basicConfig(filename=log_file, level=logging.INFO, format='[%(levelname)s:%(filename)s:%(lineno)d] %(message)s')
        self.note_type = note_type

        self.crochet: int = None
        self._tree: ElementTree
        self._processed_tree: ElementTree

    @staticmethod
    def altering(key: int = -3) -> Callable[[Literal['C', 'D', 'E', 'F', 'G', 'A', 'B']], Literal['-1', '0', '1']]:
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

    @staticmethod
    def remove_temp() -> None:
        if os.path.exists(MuseScore._temp_path):
            os.remove(MuseScore._temp_path)
            logging.info(f'removed {MuseScore._temp_path!r}')
        else:
            logging.warning(f'tried to remove {MuseScore._temp_path!r} but not found')

    @property
    def subdivision(self) -> None:
        return round(4 * self.crochet * self.note_type)
    
    def _get_crochet_duration(self) -> None:
        root = self._tree.getroot()
        for part in root:
            if part.tag != 'part': continue
            for measure in part:
                for note in measure:
                    if note.tag != 'note': continue
                    note_type = note.find('type')
                    if note_type is None: continue
                    if note_type.text != 'quarter': continue
                    if note.find('time-modification') is not None: continue
                    self.crochet = int(note.find('duration').text)
                    logging.info(f'{self.crochet=}')
                    return

    def _keep_note_max_measure(self, max_subdivision: int, duration: int, duration_elem: Element) -> bool:
        duration_offset = 0 if duration_elem is None else int(duration_elem.text)
        return max_subdivision == -1 or duration - (duration_offset) <= max_subdivision * self.subdivision

    def _export(self, musicxml_file: str, to_file: str, dpi: int = 300) -> None:
        os.system(f'MuseScore3.exe {musicxml_file} --export-to {to_file} -r {dpi}')
        logging.info(f'exported {musicxml_file=!r} to {to_file=!r} with {dpi=}')

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

            attributes = measure.find('attributes')
            if attributes is not None:
                fifths = attributes.find('key/fifths')
                if fifths is not None:
                    logging.debug(f'measure {n}, new_key={fifths.text}')
                    if n <= max_measure:
                        altering = self.altering(int(fifths.text))
                    else:
                        attributes.remove(attributes.find('key'))
                
                clef = attributes.find('clef')
                if clef is not None: attributes.remove(clef)

                time_elem = attributes.find('time')
                if time_elem is not None:
                    time_elem.attrib.update({'print-object': 'no'})

            if n < max_measure: continue  # keep in score

            for direction in measure:
                if direction.tag != 'direction': continue

                direction_type = direction.find('direction-type')
                for t in self._removing_direction:
                    dtype = direction_type.find(t)
                    if dtype is not None:
                        direction_type.remove(dtype)
                if len(direction_type) == 0: measure.remove(direction)

            duration, time_mod = 0, 0
            inside_tuplet: bool = False
            keep_first_tuplet: bool = False
            remove_notes: list[Element] = []
            for note in measure:
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
                            remove_notes.append(note)
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
                if note.find('rest') is not None: continue

                for t in self._removing_note:
                    elems = note.findall(t)
                    for e in elems: note.remove(e)

                stem = note.find('stem')
                if stem is not None: stem.text = 'none'
                
                step_text = note.find('pitch/step').text
                alter = note.find('pitch/alter')
                if alter is None:
                    new_alter = Element('alter')
                    new_alter.text = altering(step_text)
                    note.find('pitch').append(new_alter)
                else:
                    alter.text = altering(step_text)
            
            for note in remove_notes:
                measure.remove(note)
            
        self._processed_tree = ElementTree(root)
        logging.info(f'parsed to tree')

    def _write_musicxml(self, musicxml_file: str) -> None:
        self._processed_tree.write(musicxml_file, encoding='UTF-8', xml_declaration=True)
        logging.info(f'wrote tree to {musicxml_file=!r}')
    
    def convert(self, to_file: str, max_measure: int, max_subdivision: int = -1, dpi: int = 300, delete_temp: bool = True, *, root: Element | None = None, single_page: bool = False) -> None:
        self._process(max_measure, max_subdivision, root=root)
        self._write_musicxml(self._temp_path)
        self._export(self._temp_path, to_file, dpi=dpi)

        if single_page:
            to_file_split = os.path.splitext(to_file)
            if os.path.exists(to_file):
                logging.info(f'overriding {to_file!r}')
                os.remove(to_file)
            os.rename(f'{to_file_split[0]}-1' + to_file_split[1], to_file)
        if delete_temp: self.remove_temp()

    def read_musicxml(self, musicxml_file: str) -> None:
        self._tree = ETree.parse(musicxml_file)
        self._get_crochet_duration()

    def get_frames(self, directory: str = 'frames', dpi: int = 300, single_page: bool = False) -> None:
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

        for measure, sub_num in zip(part, subdivisions):
            n = int(measure.attrib['number'])
            for sub in range(sub_num):
                logging.info(f'creating frame with max_measure={n}, max_subdivision={sub} out of a total of {len(part)=} measures')
                self.convert(f'{directory}\\m{n:03d}s{sub:03d}.png', n, sub, dpi=dpi, delete_temp=False, root=copy.deepcopy(root), single_page=single_page)
        
        self.remove_temp()


def main():
    mscore = MuseScore(NOTE_LENGTH)
    mscore.read_musicxml(MUSICXML_IN)
    mscore.get_frames(DIRECTORY_OUT)

if __name__ == '__main__':
    main()
