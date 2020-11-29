import argparse
from jamintune import JamInTune
import pathlib
import numpy as np
from collections import defaultdict
import pickle
import time
from pandas import DataFrame


class MedleyDBTest(object):
    NUM_SHIFTS = 3
    RANDOM_SEED = 52813

    def __init__(self, medleydbpath, exportpath=None):
        # Find the audio path of MedleyDB
        self.medleydbpath = medleydbpath
        self.audiopath = medleydbpath.joinpath("Audio")
        self.exportpath = exportpath if exportpath is not None else pathlib.Path.cwd()
        self._check_paths_validity()

        # Set up the results dict
        self.results_dict = self._get_default_results_dict()

        # Set up songmixes list
        self.songmixes = self._get_songmix_path_list()

        # Set up the randomizer and get random shifts
        self.randomizer = self._get_randomizer()
        self.rand_shifts : dict = self._get_rand_shifts()

    def _check_paths_validity(self):
        if not self.medleydbpath.is_dir():
            raise IOError("Please specify a valid folder path for MedleyDB instead of {}".format(self.medleydbpath))

        if not self.audiopath.is_dir():
            raise IOError("Error: {} not found - is this really MedleyDB?".format(self.audiopath))

        if not self.exportpath.is_dir():
            raise IOError("Please specify a valid export directory path instead of {}".format(self.exportpath))

    def _get_default_results_dict(self):
        return defaultdict(lambda: {"orig_deviation": 0.,
                                    "rand_shift_deviations": [{"shift": 0., "deviation": 0.}
                                                              for _ in range(self.NUM_SHIFTS)]})

    def _get_songmix_path_list(self):
        # Convert generator to list so that I can get its length first (otherwise I exhaust the generator).
        # Sort it just for convenience of knowing how far along I am with the test without needing index.
        return sorted(self.audiopath.glob("*/*_MIX.wav"))

    def _get_randomizer(self):
        # RandomState is fixed and does not change with numpy versions greater than 1.16.
        # That's why we're not using default_rng.
        return np.random.RandomState(self.RANDOM_SEED)

    def _get_rand_shifts(self) -> dict:
        # Get uniform random values in [-.5, -.05] U (.05, .5]
        # The following link is why (perceptible pitch difference would be like 5-6%):
        # https://sound.stackexchange.com/questions/40386/smallest-pitch-difference-audible/42866#42866
        rand_shifts = self.randomizer.rand(len(self.songmixes), self.NUM_SHIFTS)
        rand_shifts *= .9
        rand_shifts -= .5  # .9x - .5 maps [0,1) to [-.5, .4)
        rand_shifts[rand_shifts > -.05] += .1
        return dict(zip(self.songmixes, rand_shifts))  # Associate each filename to an array of shifts this way.

    def run(self):
        csvpath = self.exportpath / "test_medleydb.csv"
        # Convert generator to list so that I can get its length first (otherwise I exhaust the generator).
        for idx, songmix in enumerate(self.songmixes):
            self.test_song(songmix)
            df = DataFrame.from_dict({songmix: self.results_dict[songmix]}, orient='index')
            df.to_csv(csvpath, mode="a", header=(False if idx else True))
            # if idx == 1:
            #     break

        # (For now) Export the dictionary as pickle
        with open(self.exportpath / "test_medleydb.pkl", "wb") as outfile:
            # Get rid of defaultdict format so it can be pickled
            pickle.dump(dict(self.results_dict), outfile, pickle.HIGHEST_PROTOCOL)

    def test_song(self, songmix: pathlib.Path):
        print("Processing {}".format(songmix))

        # (1) Get the pitch deviation of the song
        jit = JamInTune.JamInTune(str(songmix))
        jit.harmonic_separator()
        self.results_dict[songmix]["orig_deviation"] = jit.eval_deviation()

        # (2) For each candidate random shift of the song, shift the song and then perform JIT on that song
        for idx, pitch_shift in enumerate(self.rand_shifts[songmix]):
            print("Pitch shift {}".format(pitch_shift))
            jit.modify_pitch(-pitch_shift)  # Negative because it's expecting the deviation you want to undo.
            jitshift = JamInTune.JamInTune(jit.output_filename)
            jitshift.harmonic_separator() ## UGH REQUIRES CLEANUP of harmonic (not to mention the tuned) files!!!
            self.results_dict[songmix]["rand_shift_deviations"][idx]["shift"] = pitch_shift
            self.results_dict[songmix]["rand_shift_deviations"][idx]["deviation"] = jitshift.eval_deviation()


def main():
    q = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("medleydbpath", help="Path to MedleyDB")
    parser.add_argument("--exportpath", help="Export path (default: this folder)")
    args = parser.parse_args()

    medleydbpath = pathlib.Path(args.medleydbpath).expanduser()
    exportpath = pathlib.Path(args.exportpath).expanduser() if args.exportpath is not None else None

    mdtest = MedleyDBTest(medleydbpath, exportpath)
    mdtest.run()

    print("Time to run: {}".format(time.time() - q))


if __name__ == "__main__":
    main()
