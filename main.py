import os


def prepare_mfccs(path):
    os.rename(path + "audio_files/", path + "wav/")
    turn_waves_to_mfcc(path, numcep=20)
    os.rename(path + "wav/", path + "audio_files/")


if __name__ == '__main__':
    path = "<path to hey_snips_fl_5.0 or hey_snips_kws_4.0>"
    # example: path = "/mnt/c/hey_snips_fl_5.0/hey_snips_fl_amt/"

    prepare_mfccs(path)
    train()

