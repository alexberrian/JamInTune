import argparse
import pickle
import time
import numpy as np
from statsmodels.stats.weightstats import _tstat_generic
import pathlib
from scipy.stats import sem
# from statsmodels.stats.power import tt_solve_power
# from scipy.stats import shapiro
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import kurtosistest
import matplotlib.pyplot as plt

class MedleyDBEval(object):
    HUMAN_ERROR = 0.03

    def __init__(self, pickle_path=None):
        self.pickle_path = pickle_path if pickle_path is not None else pathlib.Path("test_medleydb.pkl")
        self.results_dict = self._load_pickle_file()
        self.errors = []

    def _load_pickle_file(self) -> dict:
        with open(self.pickle_path, "rb") as openfile:
            tmp = pickle.load(openfile)
            if type(tmp) != dict:
                raise IOError("Failed to load dictionary from {}".format(self.pickle_path))
            else:
                return tmp

    @staticmethod
    def _get_deviation_in_range(deviation):
        if deviation > .5:
            deviation -= 1
        elif deviation <= -.5:  # Note we don't allow -.5
            deviation += 1
        return deviation

    def _calculate_deviation_errors(self):
        for songmixpath in self.results_dict:
            orig_deviation = self.results_dict[songmixpath]["orig_deviation"]
            for experiment in self.results_dict[songmixpath]["rand_shift_deviations"]:
                orig_shift_plus_deviation = self._get_deviation_in_range(orig_deviation + experiment["shift"])
                calc_deviation = self._get_deviation_in_range(experiment["deviation"])
                self.errors.append(orig_shift_plus_deviation - calc_deviation)
                break  # To only do one

    def run(self):
        self._calculate_deviation_errors()
        # self.errors = np.log10(1 + np.asarray(self.errors))
        the_mean = np.mean(self.errors)
        the_standard_error = sem(self.errors)
        the_std = np.std(self.errors, ddof=1)
        the_nobs = len(self.errors)
        the_diff = self.HUMAN_ERROR
        t, p = _tstat_generic(the_mean, 0, the_standard_error, the_nobs - 1, "smaller", diff=the_diff)

        print("T-test: t value {} and p value {}".format(t, p))
        print("Mean: {}".format(the_mean))
        print("Standard error: {}".format(the_standard_error))
        print("Standard deviation: {}".format(the_std))
        print("Population: {}".format(the_nobs))
        print("Effect size: {}".format(the_mean/the_std))
        # print("Power: {}".format(tt_solve_power(effect_size=the_mean/the_std, nobs=the_nobs,
        #                                         alpha=0.05, power=None, alternative='smaller')))
        print("Skewness: {}".format(skew(np.asarray(self.errors))))
        print("Kurtosis: {}".format(kurtosis(np.asarray(self.errors))))
        print("Kurtosis test: {}".format(kurtosistest(np.asarray(self.errors))))
        plt.hist(self.errors, bins=61)
        plt.show(block=False)
        input("Press Enter to continue")



def main():
    q = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--picklepath", help="Path to pickle file containing results")
    args = parser.parse_args()

    picklepath = args.picklepath
    eval = MedleyDBEval(picklepath)
    eval.run()

    print("Time to run: {}".format(time.time() - q))


if __name__ == "__main__":
    main()
