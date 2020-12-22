from jamintune.audio_io import AudioSignal  #TODO: Make this code depend upon AdapTFR
import numpy as np
import soundfile as sf
import weightedstats as ws
import sox
import librosa  #TODO: Replace librosa with real-time simultaneous HPSS and tuning
import os
from copy import deepcopy
from scipy.ndimage import median_filter

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
        self.wft_shape = None
        self.wft_memory = None  # Memory of WFTs used to compute the HPSS
        self.harm_memory = None  # Memory of HPSS frames used to do inverse STFT before computing RF.

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
            "buffermode":                "centered_analysis",
            "hpss_margin":                               3.0,
            "hpss_kernel_harmonic":   self.KERNEL_HARMONIC_SIZE,
            "hpss_kernel_percussive": self.KERNEL_PERCUSSIVE_SIZE,
        }
        self._check_deviation_params_valid()

    def _initialize_helpers_hpss(self, channels):
        """
        Initialize helper arrays and variables for HPSS
        :return:
        """
        fftsize = self.deviation_param_dict["fftsize"]
        wft_memory_num_frames = self.deviation_param_dict["hpss_kernel_harmonic"]
        num_bins_up_to_nyquist = (fftsize // 2) + 1
        self.wft_shape = (channels, num_bins_up_to_nyquist)
        self.wft_memory = np.zeros([wft_memory_num_frames, *self.wft_shape], dtype=complex)

    def _check_deviation_params_valid(self):
        windowsize = self.deviation_param_dict["windowsize"]
        fftsize = self.deviation_param_dict["fftsize"]
        hopsize = self.deviation_param_dict["hopsize"]
        buffermode = self.deviation_param_dict["buffermode"]
        if windowsize < 4:
            raise ValueError("windowsize {} must be at least 4 to deal with potential edge cases".format(windowsize))
        if hopsize > windowsize:
            raise ValueError("Not allowed to have hopsize {} larger than windowsize {} "
                             "because of the way SoundFile processes chunks".format(hopsize, windowsize))
        if windowsize > fftsize:
            raise ValueError("Cannot have windowsize {} larger than fftsize {}".format(windowsize, fftsize))
        if buffermode not in ["centered_analysis", "valid_analysis"]:
            raise ValueError("Invalid buffermode {}".format(buffermode))

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
    def _pad_boundary_rows(input_array: np.ndarray, finalsize: int, side: str, flip: bool = False) -> np.ndarray:
        """
        Pad each channel of a buffer, where channels are assumed to be in rows.
        Padding happens at the boundary, by even reflection.

        :param input_array: array to be padded by reflection
        :param finalsize: finalsize: final size of the array (example: window size)
        :param side: "left" or "right" to do the padding.
                     i.e., if "left", then padding is done to the left of the input array.
        :param flip: Flip the array before exporting (default False)
        :return: output_array: reflection-padded array
        """
        inputsize = input_array.shape[1]
        if finalsize == inputsize:
            return input_array
        else:
            padsize = finalsize - inputsize
            if side == "left":
                padsize_left = padsize
                padsize_right = 0
            elif side == "right":
                padsize_left = 0
                padsize_right = padsize
            else:
                raise ValueError("Pad side {} to pad_boundary_rows is invalid, "
                                 "must be 'left' or 'right'".format(side))

            if len(input_array.shape) == 2:
                output_array = np.pad(input_array, ((0, 0), (padsize_left, padsize_right)), mode='reflect')
            else:
                raise ValueError("input array to pad_boundary_rows has dimensions {}, "
                                 "which is not supported... must be 2D array even if mono".format(input_array.shape))

            if flip:
                return output_array[::-1]
            else:
                return output_array

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

    def _median_filter_harmonic(self, input_array: np.ndarray) -> np.ndarray:
        """
        Returns an array of shape (channels, frequency bins) that computes the medians in the time direction,
        with phase information discarded.
        :param input_array:
        :return:
        """
        assert(input_array.shape[0] == self.deviation_param_dict["hpss_kernel_harmonic"])
        output_shape = input_array.shape[1:]
        output_mag = np.zeros(output_shape)

        for channel in range(output_shape[0]):
            mag = np.abs(input_array[:, channel, :])
            output_mag[channel] = np.median(mag, axis=0)

        return output_mag

    def _median_filter_percussive(self, input_array: np.ndarray) -> np.ndarray:
        """
        Returns an array of shape (channels, frequency bins) that computes the medians in the frequency direction,
        with phase information discarded.
        :param input_array:
        :return:
        """
        assert(input_array.ndim == 2)  # Input expected to be a single time slice
        # Pad in the frequency direction
        output_shape = input_array.shape
        output_mag = np.zeros(output_shape)
        mag = np.abs(input_array)
        filter_size = self.deviation_param_dict["hpss_kernel_percussive"]
        # padding_amount = filter_size // 2
        # mag = np.pad(mag, ((0, 0), (padding_amount, padding_amount)), mode='symmetric')  # CHECK ME

        for channel in range(output_shape[0]):
            output_mag[channel] = median_filter(mag[channel], size=filter_size)
            # for freqbin in range(output_shape[1]):  # TODO: SLOWWWWW.  Running median using two heaps? Or just skimage?
            #     output_mag[channel, freqbin] = np.median(mag[channel, freqbin:(freqbin + filter_size)])  # CHECK ME

        return output_mag

    @staticmethod
    def _soft_mask(a: np.ndarray, b: np.ndarray, margin: float, pwr: float) -> np.ndarray:
        """
        Soft mask with a margin
        :param a:
        :param b:
        :param margin:
        :param pwr:
        :return:
        """
        assert(a >= 0)
        assert(b >= 0)
        apwr = a ** pwr
        mbpwr = (margin * b) ** pwr
        divisor = apwr + mbpwr
        # Avoid bad float errors
        divisor[divisor < np.finfo(np.float32).tiny] = 1
        return apwr / divisor

    def _eval_deviation_single_frame(self, f: np.ndarray, f_plus: np.ndarray, window: np.ndarray,
                                     fftsize: int, samplerate: int, energy_constant: float, eps_logmag: float,
                                     rf_lower_bound: float, rf_upper_bound: float, log_cutoff_freqbin: float):

        wft = self._wft(f, window, fftsize)  # Calculate windowed fft of signal
        wft_plus = self._wft(f_plus, window, fftsize)  # Calculate windowed fft of shifted signal
        logabswft = np.log10(np.abs(wft) + eps_logmag)

        # Calculate reassignment frequencies (unit: normalized frequency) and deal with edge cases
        # Threshold the logabswft
        rf = self._calculate_rf(wft, wft_plus)
        in_bounds = np.where((rf >= rf_lower_bound) & (rf <= rf_upper_bound) & (logabswft >= log_cutoff_freqbin))
        logabswft = logabswft[in_bounds]
        rf = rf[in_bounds]
        magwftsq = np.power(10., 2 * logabswft)

        # Now calculate the deviations from the nearest piano key and get weights, then append weighted median
        # Note that I do multiply by samplerate/self.A0_FREQUENCY,
        # instead of division by rf_lower_bound, just in case rf_lower_bound gets changed.
        rf_logarithmic = 12 * np.log2(rf * samplerate / self.A0_FREQUENCY)
        nearest_piano_keys = np.round(rf_logarithmic).astype(int)
        deviations = rf_logarithmic - nearest_piano_keys

        if np.size(deviations):
            median = ws.numpy_weighted_median(deviations, weights=magwftsq)  # Mag-squared works, log-mag doesn't
            logenergy = np.log10((np.linalg.norm(wft) ** 2.0 / energy_constant) + eps_logmag)
            return median, logenergy
        else:
            return None

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

    def eval_deviation_in_house(self):
        # Initialize windowing variables
        hopsize = self.deviation_param_dict["hopsize"]
        windowsize = self.deviation_param_dict["windowsize"]
        windowfunc = self.deviation_param_dict["windowfunc"]
        fftsize = self.deviation_param_dict["fftsize"]
        buffermode = self.deviation_param_dict["buffermode"]
        windowsize_p1 = windowsize + 1
        halfwindowsize = windowsize // 2
        overlap = windowsize_p1 - hopsize
        window = windowfunc(windowsize)
        margin = self.deviation_param_dict["hpss_margin"]
        energy_constant = windowsize * np.linalg.norm(window)**2.0
        wft_memory_num_frames = self.deviation_param_dict["hpss_kernel_harmonic"]

        # Read the input signal
        sig = AudioSignal(self.input_filename)
        samplerate = sig.samplerate
        channels = sig.channels
        assert(channels == self.wft_memory.shape[1])
        assert(wft_memory_num_frames == self.wft_memory.shape[0])

        # Just in case the audio signal has already been read out
        sig.seek(frames=0)

        # TODO: Should I treat boundary frames too, to make HPSS buffer?  Check after done with this step.

        # Counting parameters
        wft_memory_full = False
        wft_memory_idx = -wft_memory_num_frames // 2  # (Ex: if wft_memory_num_frames == 31 then this is -16.)
        harmonic_memory_idx = -wft_memory_num_frames // 2

        # This is more complicated than I thought.
        # I have to do an inverse STFT to get the original signal back and then compute RF.
        # I could do an approximation, but it wouldn't be as accurate.
        # Note: Earliest sample v needed is the one to compute STFT frame n = -W/(2H) + 1,
        # which implies v = H - W/2 - W/2 = H - W (because need f[v] for v = 0 - n*H - W/2 BEFORE ZERO PADDING).
        # In our case v = 2048 - 8192 = -6144, so we need to reflect 6144 samples.
        # Unfortunately I will have to introduce a function different from _pad_boundary_rows
        # or revise it to cover this case (it makes assumption of more non-reflected samples than reflected ones.)

        # Left boundary treatment
        if buffermode == "reconstruction":
            initial_block = sig.read(frames=windowsize + 1, always_2d=True)  # In this case I really don't need +1.
            initial_block = initial_block.T
            # Pad the boundary with reflected audio frames, then compute wfts and do HPSS
            # CHANGE LINE BELOW
            frame0 = hopsize - windowsize
            while frame0 < 0:
                if frame0 < -(windowsize // 2):  # Boundary frames for perfect reconstruction
                    reflect_block = self._pad_boundary_rows(initial_block[:, :(-frame0)],
                                                            windowsize, 'left', flip=True)
                else:
                    reflect_block = self._pad_boundary_rows(initial_block[:, :(frame0 + windowsize)], windowsize,
                                                            'left')
                self.wft_memory[wft_memory_idx] = self._wft(reflect_block, window, fftsize)
                wft_memory_idx += 1

                if not wft_memory_full:
                    # Check whether we've wrapped around to the start of the self.wft_memory buffer
                    if wft_memory_idx == 0:
                        wft_memory_full = True
                    else:
                        # Insert the reflection in the array.
                        # Need to subtract one to get the indexing right.
                        # So the array fills from the center out,
                        # and the furthest boundary frames get overwritten first.
                        self.wft_memory[-wft_memory_idx - 1] = deepcopy(self.wft_memory[wft_memory_idx])

                if wft_memory_full:  # Compute HPSS
                    # Make sure the index rolls over
                    wft_memory_idx %= wft_memory_num_frames

                    # Do median filtering
                    harmonic_array = self._median_filter_harmonic(self.wft_memory)
                    percussive_array = self._median_filter_percussive(self.wft_memory[harmonic_memory_idx])

                    # Soft mask here to yield the final harmonic spectrum (with margin parameter etc.)
                    harmonic_mask = self._soft_mask(harmonic_array, percussive_array, margin=margin, pwr=2)
                    self.harm_memory[wft_memory_idx] = harmonic_mask * self.wft_memory[harmonic_memory_idx]

                frame0 += hopsize

        elif buffermode == "centered_analysis":
            initial_block = sig.read(frames=windowsize + 1, always_2d=True)  # So that RF can be computed.
            initial_block = initial_block.T
            # Pad the boundary with reflected audio frames, then compute wfts and do HPSS
            frame0 = -(windowsize // 2)  # if window is odd, this centers audio frame 0.
            while frame0 < 0:
                reflect_block = self._pad_boundary_rows(initial_block[:, :(frame0 + windowsize)], windowsize, 'left')
                self.wft_memory[wft_memory_idx] = self._wft(reflect_block, window, fftsize)
                wft_memory_idx += 1

                if not wft_memory_full:
                    # Check whether we've wrapped around to the start of the self.wft_memory buffer
                    if wft_memory_idx == 0:
                        wft_memory_full = True
                    else:
                        # Insert the reflection in the array.
                        # Need to subtract one to get the indexing right.
                        # So the array fills from the center out,
                        # and the furthest boundary frames get overwritten first.
                        self.wft_memory[-wft_memory_idx - 1] = deepcopy(self.wft_memory[wft_memory_idx])

                if wft_memory_full:  # Compute HPSS
                    # Make sure the index rolls over
                    wft_memory_idx %= wft_memory_num_frames

                    # Do median filtering
                    harmonic_array = self._median_filter_harmonic(self.wft_memory)
                    percussive_array = self._median_filter_percussive(self.wft_memory[harmonic_memory_idx])

                    # Soft mask here to yield the final harmonic spectrum (with margin parameter etc.)
                    harmonic_mask = self._soft_mask(harmonic_array, percussive_array, margin=margin, pwr=2)
                    self.harm_memory[wft_memory_idx] = harmonic_mask * self.wft_memory[harmonic_memory_idx]

                frame0 += hopsize

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
        buffermode = self.deviation_param_dict["buffermode"]
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

        # Just in case the audio signal has already been read out
        sig.seek(frames=0)

        # Left boundary treatment
        if buffermode == "centered_analysis":
            initial_block = sig.read(frames=windowsize_p1, always_2d=True)
            initial_block = initial_block.T
            # Pad the boundary with reflected audio frames, then get the medians, logenergies from those frames
            frame0 = -(windowsize // 2)  # if window is odd, this centers audio frame 0.
            while frame0 < 0:
                reflect_block = self._pad_boundary_rows(initial_block[:, :(frame0 + windowsize)], windowsize, 'left')
                reflect_block_plus = self._pad_boundary_rows(initial_block[:, :(frame0 + windowsize_p1)],
                                                             windowsize, 'left')
                output = self._eval_deviation_single_frame(reflect_block, reflect_block_plus, window, fftsize,
                                                           samplerate, energy_constant, eps_logmag,
                                                           rf_lower_bound, rf_upper_bound, log_cutoff_freqbin)
                if output is not None:
                    median, logenergy = output
                    medians_per_stft_frame.append(median)
                    logenergies_per_stft_frame.append(logenergy)
                frame0 += hopsize
        elif buffermode == "valid_analysis":
            frame0 = 0
        else:
            raise ValueError("Invalid buffermode {}".format(buffermode))

        # Get the number of audio frames, and seek to the the first audio frame
        num_audio_frames = sig.get_num_frames_from_and_seek_start(start_frame=frame0)

        # Now calculate the max number of FULL non-boundary frames you need to compute RF,
        # considering hop size and window size.
        num_full_rf_frames = 1 + ((num_audio_frames - windowsize_p1) // hopsize)

        # Convert that to the number of audio frames that you'll analyze for non-boundary RF.
        num_audio_frames_full_rf = (num_full_rf_frames - 1) * hopsize + windowsize_p1

        # Feed blocks to create the non-boundary RF frames
        blockreader = sig.blocks(blocksize=windowsize_p1, overlap=overlap,
                                 frames=num_audio_frames_full_rf, always_2d=True)

        np.set_printoptions(threshold=np.inf)

        for block in blockreader:
            block = block.T                                         # First transpose to get each channel as a row
            output = self._eval_deviation_single_frame(block[:, :windowsize], block[:, 1:], window, fftsize,
                                                       samplerate, energy_constant, eps_logmag,
                                                       rf_lower_bound, rf_upper_bound, log_cutoff_freqbin)
            if output is not None:
                median, logenergy = output
                medians_per_stft_frame.append(median)
                logenergies_per_stft_frame.append(logenergy)
            frame0 += hopsize

        # Right boundary treatment
        if buffermode == "centered_analysis":
            # Need to read from frame0
            sig.seek(frames=frame0)
            # Read the rest of the file (length less than windowsize+1)
            final_block = sig.read(always_2d=True)
            final_block = final_block.T
            final_block_num_frames = final_block.shape[1]
            if final_block_num_frames >= windowsize_p1:
                raise ValueError("You shouldn't have final_block_num_frames {} "
                                 "greater than windowsize + 1 == {}".format(final_block_num_frames, windowsize_p1))
            # Pad the boundary with reflected audio frames,
            # then add boundary SST frames to the SST
            frame1 = 0
            halfwindowsize = (windowsize + 1) // 2   # Edge case: odd windows, want final valid sample to be in middle
            while final_block_num_frames - frame1 >= halfwindowsize:
                reflect_block = self._pad_boundary_rows(final_block[:, frame1:], windowsize, 'right')
                reflect_block_plus = self._pad_boundary_rows(final_block[:, frame1 + 1:(frame1 + windowsize_p1)],
                                                             windowsize, 'right')
                output = self._eval_deviation_single_frame(reflect_block, reflect_block_plus, window, fftsize,
                                                           samplerate, energy_constant, eps_logmag,
                                                           rf_lower_bound, rf_upper_bound, log_cutoff_freqbin)
                if output is not None:
                    median, logenergy = output
                    medians_per_stft_frame.append(median)
                    logenergies_per_stft_frame.append(logenergy)
                frame1 += hopsize
        elif buffermode == "valid_analysis":  # Do nothing at this point
            pass
        else:
            raise ValueError("Invalid buffermode {}".format(buffermode))

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

    def jam_out(self, direction="closest", bias=0, verbose=False):
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
        if verbose:
            print("Deviation: {}".format(deviation))
        self.modify_pitch(deviation, direction=direction, bias=bias)



