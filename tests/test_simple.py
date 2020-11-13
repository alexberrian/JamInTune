from jamintune import JamInTune

filename = "/home/rejinal/mylastbreath.wav"
jam = JamInTune.JamInTune(filename)
jam.jam_out(direction="down", verbose=True)