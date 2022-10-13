from animate import MuseScore

if __name__ == '__main__:
  mscore = MuseScore(1/16)
    mscore.read_musicxml('chopin_op10_no4.musicxml')
    mscore.get_frames('frames', single_page=True, dpi=187)
