import os
import logging
import xml.etree.ElementTree as ETree
from xml.etree.ElementTree import ElementTree


class MuseScore:
    _removing_direction: set = {'dynamics', 'words', 'octave-shift', 'pedal', 'wedge'}
    _removing_note: list = ['accidental', 'beam', 'notations']

    def __init__(self, log_file: str | None = None) -> None:
        logging.basicConfig(filename=log_file, level=logging.INFO, format='[%(levelname)s:%(filename)s:%(lineno)d] %(message)s')
        self._tree: ElementTree

    def _read_musicxml(self, musicxml_file: str, max_measure: int) -> None:
        root = ETree.parse(musicxml_file).getroot()
        for branch in root:
            if branch.tag == 'part':
                part = branch
                logging.info(f'found part with id={part.attrib["id"]!r}')
                break
        else:
            logging.error(f'could not find \'part\' in {musicxml_file=!r}')
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
                    attributes.remove(attributes.find('key'))
                
                clef = attributes.find('clef')
                if clef is not None: attributes.remove(clef)

                time = attributes.find('time')
                if time is not None: time.attrib.update({'print-object': 'no'})

            if n <= max_measure: continue

            for direction in measure:
                if direction.tag != 'direction': continue

                direction_type = direction.find('direction-type')
                for t in self._removing_direction:
                    dtype = direction_type.find(t)
                    if dtype is not None:
                        direction_type.remove(dtype)
                if len(direction_type) == 0: measure.remove(direction)

            for note in measure:
                if note.tag != 'note': continue
                note.attrib.update({'print-object': 'no'})
                if note.find('rest') is not None: continue

                for t in self._removing_note:
                    elem = note.find(t)
                    if elem is not None: note.remove(elem)

                stem = note.find('stem')
                if stem is not None: stem.text = 'none'

                alter = note.find('pitch/alter')
                if alter is not None: note.find('pitch').remove(alter)
            
        self._tree = ElementTree(root)
        logging.info(f'parsed {musicxml_file=!r} to tree')

    def _write_musicxml(self, musicxml_file: str) -> None:
        self._tree.write(musicxml_file, encoding='UTF-8', xml_declaration=True)
        logging.info(f'wrote tree to {musicxml_file=!r}')

    def _export(self, musicxml_file: str, to_file: str, dpi: int = 300) -> None:
        os.system(f'MuseScore3.exe {musicxml_file} --export-to {to_file} -r {dpi}')
        logging.info(f'exported {musicxml_file=!r} to {to_file=!r} with {dpi=}')
    
    def convert(self, musicxml_file: str, to_file: str, max_measure: int, dpi: int = 300, delete_temp: bool = True) -> None:
        self._read_musicxml(musicxml_file, max_measure)
        self._write_musicxml('__temp__.musicxml')
        self._export('__temp__.musicxml', to_file=to_file, dpi=dpi)

        to_file_split = os.path.splitext(to_file)
        if os.path.exists(to_file): os.remove(to_file)
        os.rename(f'{to_file_split[0]}-1' + to_file_split[1], to_file)  # Musescore seems to append '-1' to the filename (probably part number) and this is a dirty fix
        if delete_temp:
            os.remove('__temp__.musicxml')
            logging.info('removed \'__temp__.musicxml\'')


if __name__ == '__main__':
    mscore = MuseScore()
    mscore.convert(MUSICXML_IN, FILE_OUT, MAX_MEASURES)
