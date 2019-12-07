import glob
import os

import numpy as np
from python_speech_features import mfcc
import progressbar
import scipy.io.wavfile as wav


def turn_waves_to_mfcc(path_to_waves, numcep=13):
    path_to_mfcc = make_sure_mfcc_path_exists(path_to_waves)
    wave_files_paths = glob.glob(path_to_waves + "/wav/*.wav")
    for idx, wave_path in enumerate(progressbar.progressbar(wave_files_paths)):
        mfcc_path = path_to_mfcc + "/" + wave_path.split("/")[-1].replace(".wav", ".mfcc")
        if not os.path.exists(mfcc_path + ".npy"):
            (rate, sig) = wav.read(wave_path)
            if len(sig) > 0:
                mfcc_feat = mfcc(sig, rate, numcep=numcep)
                np.save(mfcc_path, mfcc_feat)


def make_sure_mfcc_path_exists(path_to_audio):
    paths_of_path = path_to_audio.split("/")
    path_to_mfcc = "/".join(paths_of_path[:-1] + ["mfcc"])
    if not os.path.exists(path_to_mfcc):
        os.mkdir(path_to_mfcc)
    return path_to_mfcc
