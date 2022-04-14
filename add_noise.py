
"""in the original implimentation, the author start a new array as zeros,
and then add perturbations to the zeros array, to show the spectral importance.
Here, we use the same original data nad only perturb the required spectrum 
one at a time"""

import numpy as np
from scipy import fftpack
import copy

def addDataNoise(origSignals,band=[],channels=[],srate=100, zeroing=False):
    np.random.seed(seed=404)
    signals = copy.deepcopy(origSignals)

    if (len(band)+len(channels)) == 0:
        return origSignals
    
    if (len(channels)>0) and (len(band)==0):
        for s in range(len(signals)):
            for c in channels:
                cleanSignal = origSignals[s][c,:]
                timeDomNoise = np.random.normal(np.mean(cleanSignal), 
                                                np.std(cleanSignal), 
                                                size=len(cleanSignal))
                signals[s][c,:] = np.float32(timeDomNoise)
                """add noise to all channels: cleanSignal + timeDomNoise"""

    if (len(band) == 2) and (type(band[0]) == int):
        if len(channels)==0:
            channels = range(signals[0].shape[0])
        numSamples = signals[0].shape[1]
        W = fftpack.rfftfreq(numSamples,d=1./srate)
        lowHz = next(x[0] for x in enumerate(W) if x[1] > band[0])
        highHz = next(x[0] for x in enumerate(W) if x[1] > band[1])
        for s in range(len(signals)):
            for c in channels: #loop through channels
                dataDFT = fftpack.rfft(origSignals[s][c,:])
                cleanDFT = dataDFT[lowHz:highHz]
                freqDomNoise = np.random.normal(np.mean(cleanDFT), 
                                                np.std(cleanDFT), 
                                                size=len(cleanDFT))
                dataDFT[lowHz:highHz] =  freqDomNoise#cleanDFT + freqDomNoise
                signals[s][c,:] = np.float32(fftpack.irfft(dataDFT))


    elif (len(band)>0) and (type(band) == list):
        if len(channels)==0:
            channels = range(origSignals[0].shape[0])
        
        numSamples = origSignals[0].shape[1]
        W = fftpack.rfftfreq(numSamples, d=1./srate)    
        
        for s in range(len(signals)):
            for c in channels: #loop through channels
                dataDFT_original = fftpack.rfft(origSignals[s][c,:])
                dataDFT_output = fftpack.rfft(signals[s][c,:])
                for b in band:
                    lowHz = next(x[0] for x in enumerate(W) if x[1] > b[0])
                    highHz = next(x[0] for x in enumerate(W) if x[1] > b[1])
                    cleanDFT = dataDFT_original[lowHz:highHz]
                    freqDomNoise = np.random.normal(np.mean(cleanDFT), 
                                                    np.std(cleanDFT), 
                                                    size=len(cleanDFT))
                    if zeroing:
                        dataDFT_output[lowHz:highHz] =  0 #no signal in this frequency
                    else:
                        dataDFT_output[lowHz:highHz] = freqDomNoise #freqDomNoise
                signals[s][c,:] = np.float32(fftpack.irfft(dataDFT_output))

    return signals
