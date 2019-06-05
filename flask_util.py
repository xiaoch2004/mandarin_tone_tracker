# -*- coding: utf-8 -*-

import os
import numpy as np
import crepe
import base64


from scipy.io.wavfile import read
INT16_FAC = (2**15)-1
INT32_FAC = (2**31)-1
INT64_FAC = (2**63)-1
norm_fact = {'int16': INT16_FAC, 'int32': INT32_FAC,
             'int64': INT64_FAC, 'float32': 1.0, 'float64': 1.0}


def wavread(filename):
    """
    Read a sound file and convert it to a normalized floating point array
    filename: name of file to read
    returns fs: sampling rate of file, x: floating point array
    """

    if (os.path.isfile(filename) is False):
        # raise error if wrong input file
        raise ValueError("Input file is wrong")

    fs, x = read(filename)

    if (len(x.shape) != 1):
        # raise error if more than one channel
        raise ValueError("Audio file should be mono")

    # convert audio into floating point number in range of -1 to 1
    x = np.float32(x)/norm_fact[x.dtype.name]
    return fs, x


def readwavtobinary(filename):
    in_file = open(filename, "rb")
    data = in_file.read()
    in_file.close()
    return data


def b64wavread(b64):
    """
    Read a wav file to numpy array from b64 string
    """
    binary = base64.b64decode(b64)
    result = np.frombuffer(binary, dtype=np.dtype('int16'))
    x = result[22:]
    x = np.float32(x)/norm_fact[x.dtype.name]
    return x


def getFrameEnergy(time, x, sr):
    timestamps = time*sr
    timestamps = timestamps.astype(int)
    timestamps = np.append(timestamps, x.size-1)
    energy = [np.mean(abs(x[timestamps[i]:timestamps[i+1]])**2)
              for i in range(timestamps.size-1)]
    return np.array(energy)


def preProcess(x, sr, time, freq, confi):
    if (time.size != freq.size) or (time.size != confi.size):
        raise ValueError(
            'time, frequency and confidence array sizes does not match')
    confidenceThreshold = 0.30
    target_mean = 300
    energyThreshold = 5e-4
    # Abandon low energy values
    E = getFrameEnergy(time, x, sr)
    freq[np.where(E < energyThreshold)] = np.nan
    # Abandon low confidence values
    freq[np.where(confi < confidenceThreshold)] = np.nan
    # Abandon low frequency values
    freq[np.where(freq < 100.0)] = np.nan
    freq[np.where(freq > 700.0)] = np.nan
    # Adjust the mean
    mean = np.nanmean(freq)
    diff = target_mean - mean
    freq = freq + diff
    return (time, freq)


def base64WavAnal(wav_base64, sr):
    x = b64wavread(wav_base64)
    time, frequency, confidence, activation = crepe.predict(
        x, sr)
    time, frequency = preProcess(x, sr, time, frequency, confidence)
    return (time, frequency)


if __name__ == "__main__":
    file = 'audio/a2-cg2.wav'
    sr, x = wavread(file)
    x_dtype = x.dtype
    x_base64 = base64.b64encode(x)
    
    pack={'SampleRateHertz':sr, 'dtype':x_dtype, 'data':x_base64}

    time, freq = base64WavAnal(pack['data'], pack['dtype'], pack['sr'])
    
