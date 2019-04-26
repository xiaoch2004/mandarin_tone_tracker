import os
import matplotlib.pyplot as plt
import numpy as np
import crepe
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

    if (os.path.isfile(filename) == False):
        # raise error if wrong input file
        raise ValueError("Input file is wrong")

    fs, x = read(filename)

    if (len(x.shape) != 1):
        # raise error if more than one channel
        raise ValueError("Audio file should be mono")

    # convert audio into floating point number in range of -1 to 1
    x = np.float32(x)/norm_fact[x.dtype.name]
    return fs, x


def processTable(table):
    # Abandon the low-confidence parts
    confidenceLevel = 0.30
    target_mean = 300
    table[np.where(table[:, 2] < confidenceLevel), 1] = np.nan
    # Abandon the low-frequency parts
    table[np.where(table[:, 1] < 100.0), 1] = np.nan
    table[np.where(table[:, 1] > 700.0), 1] = np.nan
    # Adjust the mean
    mean = np.nanmean(table[:, 1], axis=0)
    std = np.nanstd(table[:, 1], axis=0)
    diff = 300 - mean
    table[:, 1] = (table[:, 1] + diff)
    return table


def getFrameEnergy(time, x, sr):
    timestamps = time*sr
    timestamps = timestamps.astype(int)
    timestamps = np.append(timestamps, x.size-1)
    energy = [np.mean(abs(x[timestamps[i]:timestamps[i+1]])**2)
              for i in range(timestamps.size-1)]
    print(energy)
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
    diff = 300 - mean
    freq = freq + diff
    return (time, freq)


def plotFourFigures(paths):
    plotnum = 0
    plt.figure()
    for file in paths:
        print(file)
        table = np.genfromtxt(file, delimiter=',')
        table = table[1:]
        time = table[:, 0]
        table = processTable(table)
        # if plotnum <= 4:
        #     print(table)
        #     print("mean:", np.nanmean(table[:, 1]))
        plt.subplot(2, 2, plotnum+1)
        plt.plot(time, table[:, 1])
        plt.title(file[8:-7])
        plt.xlim(time[0], time[-1])
        plt.ylim(100, 600)
        plotnum += 1
    plt.show()


def wavPitchCrepeAnal(filepath):
    sr, x = wavread(filepath)
    time, frequency, confidence, activation = crepe.predict(
        x, sr)
    return (time, frequency, confidence, x, sr)


if __name__ == "__main__":
    test = False
    dirname = 'audio/'
    files = os.listdir(dirname)
    paths = [dirname+file for file in files]
    if test:
        dirname = 'test/'
        files = os.listdir(dirname)
        paths = [dirname+file for file in files]
        plotFourFigures(paths[12:16])
    else:
        filepath = paths[1]
        print(filepath)
        time, frequency, confidence, x, sr = wavPitchCrepeAnal(filepath)
        filename = os.path.basename(filepath)
        dirname = os.path.dirname(filepath)
        time, frequency = preProcess(x, sr, time, frequency, confidence)
        output = np.column_stack((time, frequency))
        np.savetxt('result/' + filename[:-3] + 'csv', output, delimiter=',')
        plt.plot(time, frequency)
        plt.xlim(time[0], time[-1])
        plt.ylim(100, 500)
        plt.show()
