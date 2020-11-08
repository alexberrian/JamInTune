from soundfile import SoundFile
import os

"""
For audio i/o; essentially just does a filename check then gets SoundFile as a whole.
"""


class AudioSignal(SoundFile):

    def __init__(self, filename):
        self.filename = filename
        self.filename_check()
        super(AudioSignal, self).__init__(filename)

    def filename_check(self):
        if self.filename is None:
            raise IOError("Must specify filename")
        elif not os.path.exists(self.filename):
            raise IOError("File {} does not exist".format(self.filename))
        elif not os.path.isfile(self.filename):
            raise IOError("Filename {} is not a valid file (is it a folder?)".format(self.filename))

    def get_num_frames_from_and_seek_start(self, start_frame=0, end_frame=None):
        """
        Get the number of audio frames from start_frame to end_frame (latter is optional)
        and change the seek position in the audio file to start_frame.
        :param start_frame: int, where you intend to move the seek position to
        :param end_frame: int or None, final frame up to which you want to calculate number of frames
        :return: num_frames: number of frames between start_frame and end_frame or end of signal
        """
        return self._prepare_read(start=start_frame, stop=end_frame, frames=-1)
