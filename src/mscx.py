from xml.etree.ElementTree import XMLParser, Element, TreeBuilder, parse as parse_etree
from collections.abc import Iterator

CHORD_SUB: list[str] = ['Stem', 'NoteDot', 'Note', 'Hook']
GRACENOTE: list[str] = {'grace4', 'acciaccatura', 'appoggiatura', 'grace8after', 'grace16', 'grace16after', 'grace32', 'grace32after'}
INVISIBLITY_ALLOWED: set[str] = {'Accidental', 'Rest'}
UNPRINTABLE: set[str] = {'stretch', 'startRepeat', 'endRepeat', 'MeasureNumber', 'LayoutBreak', 'vspacerUp', 'vspacerDown', 'vspacerFixed'}


class ElementTagError(Exception):
    pass


class MElement(Element):
    def __init__(self, tag: str, attrib: dict[str, str] = ..., **extra: str) -> None:
        super().__init__(tag, attrib, **extra)
        self.protected: bool = False

    def _visible(self) -> tuple['MElement', bool]:
        visible = self.find('visible')
        if visible is None:
            return None, True
        if visible.text != '0':
            return visible, True
        else:
            return visible, False

    def protect(self) -> None:
        self.protected = True

    def get_chord_subelements(self) -> Iterator['MElement']:
        if self.tag != 'Chord':
            raise ElementTagError(f"MElement has tag {self.tag!r}, should be 'Chord': cannot get chord elements")
        for tag in CHORD_SUB:
            for element in self.iter(tag):
                yield element

    def get_tuplet(self) -> float:
        if self.tag == 'Tuplet':
            return int(self.find('normalNotes').text)/int(self.find('actualNotes').text)
        elif self.tag == 'endTuplet':
            return 1.0
        else:
            raise ElementTagError(f"MElement has tag {self.tag!r}, should be 'Tuplet' or 'endTuplet': cannot get tuplet value")

    def is_invisiblity_allowed(self) -> bool:
        return self.tag in INVISIBLITY_ALLOWED

    def is_unprintable(self) -> bool:
        return self.tag in UNPRINTABLE

    def is_gracenote(self) -> bool:
        return any(subelem.tag in GRACENOTE for subelem in self)

    def is_visible(self) -> bool:
        _, _visible = self._visible()
        return _visible
    
    def set_visible(self) -> None:
        if self.protected: return

        e, _visible = self._visible()
        if not _visible:
            self.remove(e)
    
    def set_invisible(self) -> None:
        if self.protected: return

        _, _visible = self._visible()
        if _visible: return

        invisible = type(self)('visible')
        invisible.text = '0'
        self.append(invisible)
    
    def set_visible_all(self, tag: str | None = None) -> None:
        for subelement in self.iter(tag):
            subelement: MElement
            subelement.set_visible()
    
    def set_invisible_all(self, tag: str | None = None) -> None:
        for subelement in self.iter(tag):
            subelement: MElement
            subelement.set_invisible()
    
    def set_visible_chord(self) -> None:
        for element in self.get_chord_subelements():
            element.set_visible()
    
    def set_invisible_chord(self) -> None:
        for element in self.get_chord_subelements():
            element.set_invisible()


def parse_custom_etree(source: str):
    treebuilder = TreeBuilder(element_factory=MElement)
    parser = XMLParser(target=treebuilder)
    tree = parse_etree(source, parser)
    return tree
