import sys
sys.path.append("./AdapTFT")
from AdapTFT import tft
import numpy as np
import weightedstats as ws
from scipy.signal import detrend


class JamInTune(tft.TFTransformer):
    # This is perhaps a little bit of a lowpass.  May want to care about higher frequencies too.
    A0_FREQUENCY = 27.5
    MAX_PIANO_KEY_FREQUENCY = 4186.00904481
    NUM_KEYS = 88
    PIANO_KEY_FREQUENCIES = np.array([  A0_FREQUENCY,   29.13523509,   30.86770633,   32.70319566,
                                         34.64782887,   36.70809599,   38.89087297,   41.20344461,
                                         43.65352893,   46.24930284,   48.9994295 ,   51.9130872 ,
                                         55.        ,   58.27047019,   61.73541266,   65.40639133,
                                         69.29565774,   73.41619198,   77.78174593,   82.40688923,
                                         87.30705786,   92.49860568,   97.998859  ,  103.82617439,
                                        110.        ,  116.54094038,  123.47082531,  130.81278265,
                                        138.59131549,  146.83238396,  155.56349186,  164.81377846,
                                        174.61411572,  184.99721136,  195.99771799,  207.65234879,
                                        220.        ,  233.08188076,  246.94165063,  261.6255653 ,
                                        277.18263098,  293.66476792,  311.12698372,  329.62755691,
                                        349.22823143,  369.99442271,  391.99543598,  415.30469758,
                                        440.        ,  466.16376152,  493.88330126,  523.2511306 ,
                                        554.36526195,  587.32953583,  622.25396744,  659.25511383,
                                        698.45646287,  739.98884542,  783.99087196,  830.60939516,
                                        880.        ,  932.32752304,  987.76660251, 1046.5022612 ,
                                       1108.73052391, 1174.65907167, 1244.50793489, 1318.51022765,
                                       1396.91292573, 1479.97769085, 1567.98174393, 1661.21879032,
                                       1760.        , 1864.65504607, 1975.53320502, 2093.0045224 ,
                                       2217.46104781, 2349.31814334, 2489.01586978, 2637.0204553 ,
                                       2793.82585146, 2959.95538169, 3135.96348785, 3322.43758064,
                                       3520.        , 3729.31009214, 3951.06641005, MAX_PIANO_KEY_FREQUENCY])

    def __init__(self, filename):
        super(JamInTune, self).__init__(filename)

        self.deviation_param_dict = {}
        self.initialize_deviation_params()

    def initialize_deviation_params(self):
        self.deviation_param_dict = {
            "hopsize":                                  8192,    # To avoid bug
            # "hopsize":      self.AudioSignal.samplerate // 5,  # Just hop every 0.25 seconds
            "windowfunc":                         np.hanning,
            "windowsize":                               8192,
            "fftsize":                                  8192,
            "eps_logmag":                            1.0e-16,
            # "log_cutoff_dB_freqbin":                     -15,
            "log_cutoff_dB_freqbin":                     -30,
            "log_cutoff_dB_stft_frame":                  -70,
            "lower_cutoff_freq":                        200.,
        }

    def eval_deviation(self):
        """
        Calculate the weighted median weighted median deviation from piano key frequencies. :)

        Questions:
        - Is the soft thresholding really valid or should it be a hard cutoff to preserve logarithmic scale?
        :return:
        """

        # Initialize variables
        hopsize = self.deviation_param_dict["hopsize"]
        windowsize = self.deviation_param_dict["windowsize"]
        if windowsize < 4:
            raise ValueError("windowsize {} must be at least 4 to deal with potential edge cases".format(windowsize))
        windowsize_p1 = windowsize + 1
        fftsize = self.deviation_param_dict["fftsize"]
        if windowsize > fftsize:
            raise ValueError("window size {} is larger than FFT size {}!".format(windowsize, fftsize))
        windowfunc = self.deviation_param_dict["windowfunc"]
        overlap = windowsize_p1 - hopsize
        window = windowfunc(windowsize)
        energy_constant = windowsize * np.linalg.norm(window)**2.0
        frame0 = 0
        samplerate = self.AudioSignal.samplerate
        eps_logmag = self.deviation_param_dict["eps_logmag"]
        max_piano_key_frequency_normalized = self.MAX_PIANO_KEY_FREQUENCY / samplerate
        # Set upper bound on reassignment frequency
        rf_upper_bound = min([0.5, max_piano_key_frequency_normalized])  # 0.5 is Nyquist
        lower_cutoff_freq = self.deviation_param_dict["lower_cutoff_freq"]
        lower_bound_freq = max([lower_cutoff_freq, self.A0_FREQUENCY])
        rf_lower_bound = lower_bound_freq / samplerate
        logenergies_per_stft_frame = []
        medians_per_stft_frame = []
        log_cutoff_freqbin = self.deviation_param_dict["log_cutoff_dB_freqbin"] / 20
        log_cutoff_stft_frame = self.deviation_param_dict["log_cutoff_dB_stft_frame"] / 20

        # Get the number of audio frames, and seek to the the first audio frame (no boundary treatment)
        num_audio_frames = self.AudioSignal.get_num_frames_from_and_seek_start(start_frame=frame0)

        # Now calculate the max number of FULL non-boundary frames you need to compute RF,
        # considering hop size and window size.
        num_full_rf_frames = 1 + ((num_audio_frames - windowsize_p1) // hopsize)

        # Convert that to the number of audio frames that you'll analyze for non-boundary RF.
        num_audio_frames_full_rf = (num_full_rf_frames - 1) * hopsize + windowsize_p1

        print(num_audio_frames)
        print(hopsize)
        print(num_full_rf_frames)
        print(num_audio_frames_full_rf)

        # Feed blocks to create the non-boundary RF frames
        blockreader = self.AudioSignal.blocks(blocksize=windowsize_p1, overlap=overlap,
                                              frames=num_audio_frames_full_rf, always_2d=True)

        np.set_printoptions(threshold=np.inf)

        for block in blockreader:
            block = block.T                                         # First transpose to get each channel as a row
            try:
                wft = self.wft(block[:, :windowsize], window, fftsize)  # Calculate windowed fft of signal
            except ValueError:
                print(frame0)
                raise
            wft_plus = self.wft(block[:, 1:], window, fftsize)      # Calculate windowed fft of shifted signal
            # logabswft = detrend(np.log10(np.abs(wft) + eps_logmag))  # Detrend the spectrum to make it more even
            # logabswft -= np.max(logabswft)                           # To max it at 0 dB
            logabswft = np.log10(np.abs(wft) + eps_logmag)

            # Calculate reassignment frequencies (unit: normalized frequency) and deal with edge cases
            # Soft threshold the logabswft then get strictly nonnegative weights
            rf = self._calculate_rf(wft, wft_plus)
            in_bounds = np.where( (rf >= rf_lower_bound) & (rf <= rf_upper_bound) & (logabswft >= log_cutoff_freqbin))
            logabswft = logabswft[in_bounds]
            rf = rf[in_bounds]
            # print(rf)
            # print(logabswft)

            # out_of_bounds = np.where((rf < rf_lower_bound) | (rf > rf_upper_bound) | (logabswft < log_cutoff_freqbin))
            # logabswft[out_of_bounds] = log_cutoff_freqbin
            # rf[out_of_bounds] = 0
            # logabswft -= log_cutoff_freqbin  # To use as weights... soft thresholding
            magwftsq = np.power(10., 2*logabswft)

            # Now calculate the deviations from the nearest piano key and get weights, then append weighted median
            # Note that I do multiply by samplerate/self.A0_FREQUENCY,
            # instead of division by rf_lower_bound, just in case rf_lower_bound gets changed.
            rf_logarithmic = 12 * np.log2(rf * samplerate / self.A0_FREQUENCY )
            nearest_piano_keys = np.round(rf_logarithmic).astype(int)
            # try:
            #     assert( all([ (nearest_piano_keys[k, l] >= 0) &
            #                   (nearest_piano_keys[k, l] < self.NUM_KEYS) for k, l in nearest_piano_keys]))
            #
            # except AssertionError:
            #     print(nearest_piano_keys)
            #     raise
            # nearest_piano_frequencies_normalized = self.PIANO_KEY_FREQUENCIES[nearest_piano_keys] / samplerate
            deviations = rf_logarithmic - nearest_piano_keys
            # print(deviations)
            if np.size(deviations):
                # median = ws.numpy_weighted_median(deviations, weights=logabswft)
                median = ws.numpy_weighted_median(deviations, weights=magwftsq)
                # median = np.average(deviations, weights=magwftsq)
                # print(median)
                medians_per_stft_frame.append(median)
                # Now calculate the frame's log energy and append
                logenergy = np.log10((np.linalg.norm(wft) ** 2.0 / energy_constant) + eps_logmag)
                logenergies_per_stft_frame.append(logenergy)
            frame0 += hopsize

        # After looping through all blocks, soft threshold log energies and calculate the final weighted median
        # deviation
        logenergies_per_stft_frame = np.asarray(logenergies_per_stft_frame)
        out_of_bounds = np.where(logenergies_per_stft_frame < log_cutoff_stft_frame)
        logenergies_per_stft_frame[out_of_bounds] = log_cutoff_stft_frame
        logenergies_per_stft_frame -= log_cutoff_stft_frame
        return ws.numpy_weighted_median(np.asarray(medians_per_stft_frame), logenergies_per_stft_frame)

    def modify_pitch(self):
        # This is where I have to look up the best way to do this according to Serra's course
        pass

    def jam_out(self):

        pass



