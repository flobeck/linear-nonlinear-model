import sys, os, struct, wave, subprocess, scipy
from scipy.io.wavfile import write

def read_wav(wav_file):
    w = wave.open(wav_file)
    nframes = int(0.5*10**6)

    if w.getnframes() < nframes:
        return

    return struct.unpack('{n}h'.format(n=nframes), w.readframes(nframes))

def mp3_wav(mp3_file, samplingrate):
    # mp3 -> wav (mono)
    mpg123_cmd  = 'mpg123 -w "%s" -r '+ str(samplingrate) + ' -m "%s"'
    cmd = mpg123_cmd % ('temp.wav', mp3_file)
    temp = subprocess.call(cmd, shell=True)

    return read_wav('temp.wav')

def getWAV(directory):
    W = []
    samplingrate = 44100.0
    for path, dirs, files in os.walk(directory):
        for f in files:
            wavdata = None
            if f.endswith('.mp3'):
                mp3_file = os.path.join(path, f)
                wavdata = mp3_wav(mp3_file, samplingrate)
            elif f.endswith('.wav'):
                wavdata = read_wav(f)
            if wavdata is None:
                continue
            try:
                W.append(wavdata)
                if len(W) == 70:
                    return W
            except:
                continue
    return W


def normalize_write(eigensound, wavname):
    for i in range(len(eigensound)):
        fname = wavname + str(i) + '.wav'
        normalize = np.int16(eigensound[i]/np.max(np.abs(eigensound[i])) * 2**15)
        write(fname, 44100.0, normalize)
