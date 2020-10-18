# JamInTune
<b>Tune an audio recording to match standard Western piano key frequencies</b>  (Python package)
<i>Hacking Audio Music Research (HAMR) 2020</i>  
<b>Author:</b> Alex Berrian

## Motivation
Jamming along to a song is most fun when you're in tune with the song.  
If you're a pianist, you only have to worry about tuning your piano a couple of times a year.  
And that works great for jamming to most modern popular music recordings, which are generally tuned quite close to the standard piano key frequencies.

But some recordings (including some very popular ones) are tuned to frequencies that lie distinctly between those of piano keys, 
either due to limitations in recording methodologies during the time period, or because the producer liked the way it sounded.  
While guitarists and many other instrumentalists can manually tune to match the recording, this issue generally leaves pianists out in the cold.

To solve this problem, this codebase approximates the amount to which the input recording deviates from the standard Western 
piano key frequencies, and then it shifts the pitch of the recording so that the frequencies align with the piano key frequencies.

### Examples of popular music that is tuned between piano keys
```
Soundgarden - Black Hole Sun
James Newman - My Last Breath (the United Kingdom's entry in the 2020 Eurovision Song Contest)
Gotye feat. Kimbra - Somebody That I Used To Know
The Beatles - Penny Lane
Tommy Tutone - 867-5309 / Jenny
P. Diddy feat. Ginuwine, Loon, Mario Winans, and Tammy Ruggieri - I Need a Girl (Part 2)
Derek & The Dominos - Layla
The Magnetic Fields - I Don't Believe in the Sun
Michael Jackson - Baby Be Mine
Robert Johnson - Kind Hearted Woman Blues
```
### Why use JamInTune?
- Jam to your favorite out-of-key song!
- Pitch shift old recordings for sampling in a song you make
- Make your DJ mixes sound better

## Dependencies
This code requires `python3.7` and the following Python packages, installable via pip:
```
librosa==0.8.0
soundfile==0.10.3.post1
weightedstats==0.4.1
sox==1.4.1
```

## Acceptable Inputs
This code will take in and write out whatever [`soundfile`](https://pypi.org/project/SoundFile/) can read as audio. 
However, I have only tested with `.wav` formats due to time constraints.

## Usage
Simple example:
```
import JamInTune
filename = "/path/to/blackholesun.wav"
jam = JamInTune.JamInTune(filename)
jam = jam_out()  # Exports the tuned file as "/path/to/blackholesun_tuned.wav"
```
If you want the recording to be shifted to the closest piano keys up or down, rather than the closest keys in whatever direction:
```
jam = jam_out(direction="up")  # Or direction="down"
```
You could also <i>detune</i> the recording from piano keys by using the `bias` parameter (in units of semitones):
```
jam = jam_out(bias=1.337)  # Shifts the tuned recording 1.337 semitones up
```
Or use the bias parameter to shift additional semitones if you want lower or higher keys than the closest one:
```
jam = jam_out(bias=-5)  # Shifts the tuned recording 5 semitones down, i.e. 5 piano keys down
```

## How It Works
The code works in three steps:

### 1) Isolate the harmonic content in the signal
In this step, we use the <i>harmonic-percussive source separation</i> (HPSS) algorithm [[1]](#1) of Driedger, Müller, and Disch.  
Currently, the code uses the implementation found in the [`librosa`](http://librosa.org) package (but see "Issues and Future Work" below).  
We need to do this step because wideband percussive content interferes with the method in the next step.  

### 2) Calculate the approximate deviation from the piano key frequencies in the harmonic signal
The one-sentence summary: We introduce a novel technique based on calculating <i>reassignment frequencies</i> and 
their weighted median deviation from the nearest piano key frequencies in each frame of a short-time Fourier transform (STFT), then 
calculate a weighted median of those median deviations over all STFT frames.

In detail:

- We compute the STFT of the input signal, and of the input signal shifted back in time by one sample. 
We do this with a Hann window, hop size 2048, and window/FFT size 8192. These parameters are the same as the ones used to do the HPSS.
- As each STFT frame is calculated, the <i>reassignment frequency</i> (RF) [[2](#2), [3](#3)] is calculated corresponding to each FFT bin; 
the formula to calculate each reassignment frequency resembles a finite differencing of the phase spectrum. 
(This is where the FFT of the back-shifted input signal is necessary.) 
Essentially, the RF attempts to answer the following question: "For this FFT bin, what is the actual frequency of the sinusoidal 
component closest in frequency to this bin?" (For instance, if you have a sinusoid of 449.13 Hz in the signal at this frame, 
and the nearest FFT bin is 441 Hz, then the RF of the 441 Hz bin would ideally be close to 449.13 Hz.)
- Not every RF is relevant. The ones from low magnitude FFT bins are especially irrelevant, 
and the RF is not likely to be accurate in that case anyway. 
So we discard the RF and FFT values from bins where the RF is either inaccurate 
(not between 0 and Nyquist) or the FFT magnitude is below some threshold (as of writing, -30 dB; so we are really only taking the most prominent bins).
- Next, we calculate the deviation (on logarithmic scale, in terms of semitones) of each RF from the nearest piano key frequency. This is done in O(N) time where N is the number of FFT bins.
- Using the magnitude-squared of the FFT values as weights, we calculate the weighted median of the deviations using 
the [`weightedstats`](https://pypi.org/project/weightedstats/) package, and we store this value as we continue to the next STFT frame. 
We also compute the energy in this frame by summing the magnitude-squared FFT values, and we store that value.
- Upon going to the next frame, the FFT is discarded.
- Once all frames have their weighted median deviations calculated, we threshold the frame energies by a certain value (currently, -70 dB). 
Then, using the thresholded energies as weights, we compute the weighted median (over time) of the weighted median deviations. 
This final number is our approximate deviation.
- The weighted median RF idea is an outlier-robust spinoff of the <i>weighted average RF</i> idea in [[3]](#3).

### 3) Pitch shift the recording according to the approximate deviation
We use the [`pysox`](https://pypi.org/project/sox/) Python wrapper around `sox` to pitch-shift the input recording according 
to the approximate deviation, then export it.

## Issues and Future Work
- The code runs slowly, due to repetition in the computation. 
The HPSS and deviation implementations should run concurrently (i.e., as the signal is being buffered into windowed frames), 
but there wasn't enough time to get that to happen. In other words, the spectrogram is currently double-computed.
- Unit testing is needed.
- More data evaluation is needed.

## References
<a id="1">[1]</a> 
J. Driedger, M. Müller, S. Disch. 
Extending harmonic-percussive separation of audio. 
5th International Society for Music Information Retrieval Conference (ISMIR 2014), Taipei, Taiwan, 2014.

<a id="2">[2]</a>
A. Berrian, N. Saito. 
"Adaptive synchrosqueezing based on a quilted short-time Fourier transform." 
<i>Wavelets and Sparsity XVII.</i> Vol. 10394. International Society for Optics and Photonics, 2017.

<a id="3">[3]</a> 
A. Berrian. 
"The chirped quilted synchrosqueezing transform and its application to bioacoustic signal analysis." 
Ph.D. dissertation, 2018.
