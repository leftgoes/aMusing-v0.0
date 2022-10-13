from animate import MuseScore

if __name__ == '__main__:
  mscore = MuseScore(1/16, log_file='log.txt')
    mscore.read_musicxml('chopin_etude_10_4.musicxml')
    mscore.get_frames('frames', single_page=True, dpi=187)
