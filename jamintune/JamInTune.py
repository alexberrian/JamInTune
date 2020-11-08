from jamintune.audio_io import AudioSignal  #TODO: Make this code depend upon AdapTFR
import numpy as np
import soundfile as sf
import weightedstats as ws
import sox
import librosa  #TODO: Replace librosa with real-time simultaneous HPSS and tuning
import os

TWOPI = 2 * np.pi


class JamInTune(object):
    KERNEL_HARMONIC_SIZE = 31     # For hop size 2048
    KERNEL_PERCUSSIVE_SIZE = 124  # For FFT size 8192
    # TODO: This is perhaps a little bit of a lowpass.  May want to care about higher frequencies too.
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
        self.input_filename = filename
        self.intermediate_harmonic_filename = None
        self.output_filename = None
        self.deviation_param_dict = {}
        self._initialize_deviation_params()
        self._set_intermediate_harmonic_filename()
        self._set_output_filename()

    def _initialize_deviation_params(self):
        self.deviation_param_dict = {
            "hopsize":                                  2048,
            # "hopsize":      self.AudioSignal.samplerate // 5,  # Just hop every 0.25 seconds
            "windowfunc":                         np.hanning,
            "windowsize":                               8192,
            "fftsize":                                  8192,
            "eps":                                   1.0e-16,
            # "log_cutoff_dB_freqbin":                     -15,
            "log_cutoff_dB_freqbin":                     -30,
            "log_cutoff_dB_stft_frame":                  -70,
            "lower_cutoff_freq":                        200.,  # TODO: Optimize this, preferably signal-based manner
        }
        self._check_deviation_params_valid()

    def _check_deviation_params_valid(self):
        windowsize = self.deviation_param_dict["windowsize"]
        fftsize = self.deviation_param_dict["fftsize"]
        hopsize = self.deviation_param_dict["hopsize"]
        if windowsize < 4:
            raise ValueError("windowsize {} must be at least 4 to deal with potential edge cases".format(windowsize))
        if hopsize > windowsize:
            raise ValueError("Not allowed to have hopsize {} larger than windowsize {} "
                             "because of the way SoundFile processes chunks".format(hopsize, windowsize))
        if windowsize > fftsize:
            raise ValueError("Cannot have windowsize {} larger than fftsize {}".format(windowsize, fftsize))

    def _set_intermediate_harmonic_filename(self):
        out_base, out_extn = os.path.splitext(self.input_filename)
        self.intermediate_harmonic_filename = "{}_harm{}".format(out_base, out_extn)

    def _set_output_filename(self, infile: str = None, outfile: str = None):
        if infile is None:
            infile = self.input_filename

        if outfile is None:  # Default case
            out_base, out_extn = os.path.splitext(infile)
            self.output_filename = "{}_tuned{}".format(out_base, out_extn)
        else:  # Might be good to have path validation here...
            self.output_filename = outfile

    def _calculate_rf(self, wft: np.ndarray, wft_plus: np.ndarray) -> np.ndarray:
        eps_division = self.deviation_param_dict["eps"]
        return np.angle(wft_plus / (wft + eps_division)) / TWOPI  # Unit: Normalized frequency

    def _wft(self, block: np.ndarray, window: np.ndarray, fftsize: int, fft_type="real") -> np.ndarray:
        if fft_type == "real":
            return np.fft.rfft(self._zeropad_rows(window * block, fftsize))
        elif fft_type == "complex_short":
            return np.fft.fft(self._zeropad_rows(window * block, fftsize))[:, :(1 + (fftsize // 2))]
        elif fft_type == "complex_full":  # For reconstruction
            return np.fft.fft(self._zeropad_rows(window * block, fftsize))
        else:
            raise ValueError("Invalid fft_type {}, must use 'real', 'complex_short', "
                             "or 'complex_full'".format(fft_type))

    @staticmethod
    def _zeropad_rows(input_array: np.ndarray, finalsize: int) -> np.ndarray:
        """
        Zeropad each channel of a buffer, where channels are assumed to be in rows.
        Padding happens with the input array centered, and zeros padded equally on left and right,
        unless finalsize minus inputsize is odd.
        This is used for preparing a windowed array to be sent to an FFT.

        :param input_array: array to be padded with zeros
        :param finalsize: final size of the array (example: FFT size)
        :return: output_array: zero-padded array
        """
        inputsize = input_array.shape[1]
        if inputsize == finalsize:
            return input_array
        else:
            padsize = finalsize - inputsize
            padsize_left = padsize // 2
            padsize_right = padsize - padsize_left

            if len(input_array.shape) == 2:
                output_array = np.pad(input_array, ((0, 0), (padsize_left, padsize_right)), mode='constant')
            else:
                raise ValueError("input array to zeropad_rows has dimensions {}, "
                                 "which is not supported... must be 2D array even if mono".format(input_array.shape))

            return output_array

    def harmonic_separator(self):
        """
        Creates intermediate wave file with harmonic-only content from the input wave file.
        """

        # TODO: make a circular buffer out of a numpy array to streamline this process with the rest of the code,
        # instead of doing repetitive computation and exporting an intermediate file.

        sig, sr = librosa.load(self.input_filename, sr=None)
        stft = librosa.stft(sig, n_fft=self.deviation_param_dict["fftsize"])
        harm, _ = librosa.decompose.hpss(stft, kernel_size=(31, 124), margin=3.0)
        y_harm = librosa.istft(harm)
        sf.write(self.intermediate_harmonic_filename, y_harm, sr)

    def eval_deviation(self):
        """
        Calculate the weighted median weighted median deviation from piano key frequencies. :)
        :return:
        """

        # Initialize windowing variables
        hopsize = self.deviation_param_dict["hopsize"]
        windowsize = self.deviation_param_dict["windowsize"]
        windowfunc = self.deviation_param_dict["windowfunc"]
        fftsize = self.deviation_param_dict["fftsize"]
        windowsize_p1 = windowsize + 1
        overlap = windowsize_p1 - hopsize
        window = windowfunc(windowsize)
        energy_constant = windowsize * np.linalg.norm(window)**2.0

        # Read the harmonic signal (TODO: Shouldn't have to do this, should already have harmonic spectrogram available)
        sig = AudioSignal(self.intermediate_harmonic_filename)
        samplerate = sig.samplerate
        max_piano_key_frequency_normalized = self.MAX_PIANO_KEY_FREQUENCY / samplerate

        # Set bounds on reassignment frequency
        rf_upper_bound = min([0.5, max_piano_key_frequency_normalized])  # 0.5 is Nyquist
        lower_cutoff_freq = self.deviation_param_dict["lower_cutoff_freq"]
        lower_bound_freq = max([lower_cutoff_freq, self.A0_FREQUENCY])
        rf_lower_bound = lower_bound_freq / samplerate

        # Initialize per-frame lists of log-energies and of median deviations from piano key frequencies
        logenergies_per_stft_frame = []
        medians_per_stft_frame = []

        # Set cutoffs for thresholding the log magnitudes and log energies, and epsilon for calculating log
        log_cutoff_freqbin = self.deviation_param_dict["log_cutoff_dB_freqbin"] / 20
        log_cutoff_stft_frame = self.deviation_param_dict["log_cutoff_dB_stft_frame"] / 20
        eps_logmag = self.deviation_param_dict["eps"]

        # Get the number of audio frames, and seek to the the first audio frame (no boundary treatment TODO???)
        frame0 = 0
        num_audio_frames = sig.get_num_frames_from_and_seek_start(start_frame=frame0)

        # Now calculate the max number of FULL non-boundary frames you need to compute RF,
        # considering hop size and window size.
        num_full_rf_frames = 1 + ((num_audio_frames - windowsize_p1) // hopsize)

        # Convert that to the number of audio frames that you'll analyze for non-boundary RF. (TODO???)
        num_audio_frames_full_rf = (num_full_rf_frames - 1) * hopsize + windowsize_p1

        # Feed blocks to create the non-boundary RF frames  (TODO???)
        blockreader = sig.blocks(blocksize=windowsize_p1, overlap=overlap,
                                 frames=num_audio_frames_full_rf, always_2d=True)

        np.set_printoptions(threshold=np.inf)

        for block in blockreader:
            block = block.T                                         # First transpose to get each channel as a row
            try:
                wft = self._wft(block[:, :windowsize], window, fftsize)  # Calculate windowed fft of signal
            except ValueError:
                print("Current frame at which there is an error: {}".format(frame0))
                raise
            wft_plus = self._wft(block[:, 1:], window, fftsize)      # Calculate windowed fft of shifted signal
            logabswft = np.log10(np.abs(wft) + eps_logmag)

            # Calculate reassignment frequencies (unit: normalized frequency) and deal with edge cases
            # Threshold the logabswft
            rf = self._calculate_rf(wft, wft_plus)
            in_bounds = np.where( (rf >= rf_lower_bound) & (rf <= rf_upper_bound) & (logabswft >= log_cutoff_freqbin))
            logabswft = logabswft[in_bounds]
            rf = rf[in_bounds]
            magwftsq = np.power(10., 2*logabswft)

            # Now calculate the deviations from the nearest piano key and get weights, then append weighted median
            # Note that I do multiply by samplerate/self.A0_FREQUENCY,
            # instead of division by rf_lower_bound, just in case rf_lower_bound gets changed.
            rf_logarithmic = 12 * np.log2(rf * samplerate / self.A0_FREQUENCY)
            nearest_piano_keys = np.round(rf_logarithmic).astype(int)
            deviations = rf_logarithmic - nearest_piano_keys

            if np.size(deviations):
                median = ws.numpy_weighted_median(deviations, weights=magwftsq)  # Mag-squared works, log-mag doesn't
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
        logenergies_per_stft_frame -= log_cutoff_stft_frame  # Necessary to make weights positive
        return ws.numpy_weighted_median(np.asarray(medians_per_stft_frame), logenergies_per_stft_frame)

    def modify_pitch(self, deviation: float, infile: str = None, direction: str = "closest", bias=0):
        """
        Use sox to modify the pitch.
        Exports the pitch-shifted file to self.output_filename.

        :param deviation: deviation in number of semitones (from self.eval_deviation, or choose your own)
        :param infile: put this here if you want to pitch shift a different file from self.input_filename.
        :param direction: direction in which to modify the pitch
                          "closest" (default) - pitch shifted by -deviation.
                              If the output is from self.eval_deviation, then this is trying to shift the pitch to the
                              closest standard Western music notes.
                          "up" - force shift to the nearest note up, assuming deviation gets you to standard Western
                                 music notes
                          "down" - same as above but nearest note down.
        :param bias: number of semitones up or down you want to add, can be fractional
        """
        if direction == "closest":
            pass # Do nothing
        elif direction == "up":
            if deviation > 0:  # if closest note is below
                deviation -= 1
        elif direction == "down":
            if deviation < 0:  # if closest note is above
                deviation += 1
        else:
            raise ValueError("Invalid direction {}, must be 'closest' (default), 'up', or 'down'".format(direction))
        if infile is None:
            infile = self.input_filename
        else:
            self._set_output_filename(infile=infile)

        shift = -deviation + bias

        soxtfm = sox.Transformer()
        soxtfm.pitch(shift)
        soxtfm.build_file(infile, self.output_filename)

    def jam_out(self, direction="closest", bias=0):
        """
        Jam out!
        Isolate the harmonic content from the input audio, calculate the deviation from standard Western music notes,
        and pitch-shift accordingly.
        :param direction: direction in which to modify the pitch
                          "closest" (default) - pitch shifted by -deviation.
                              If the output is from self.eval_deviation, then this is trying to shift pitches to the
                              closest standard Western music notes.
                          "up" - force shift to the nearest note up, assuming deviation gets you to standard Western
                                 music notes
                          "down" - same as above but nearest note down.
        :param bias: number of semitones up or down you want to add, can be fractional
        :return:
        """
        self.harmonic_separator()
        deviation = self.eval_deviation()
        self.modify_pitch(deviation, direction=direction, bias=bias)



