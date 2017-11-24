import numpy as np
from python_speech_features import mfcc


def preemphasis(x):
    A = 0.975
    # x has to be a numpy array!
    return x[1:] - A * x[:-1]

def audio2feature(audio,rate):
    audio=preemphasis(audio)
    mfcc_feat = mfcc(audio, rate, numcep=23, appendEnergy=False)
    avg_feature = np.average(mfcc_feat, axis=0)
    return avg_feature
