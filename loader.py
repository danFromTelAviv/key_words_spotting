import json
import os
import random

import numpy as np


class Loader_HeySnips:
    def __init__(self, json_path, mfccs_path, batch_size=32):
        self.snips_json = json.load(open(json_path, "r"))
        self.samples_json = [{"mfcc_path": mfccs_path + item["id"] + ".mfcc.npy", "label": item["is_hotword"]}
                             for item in self.snips_json]
        self.samples_json = [item for item in self.samples_json if os.path.exists(item["mfcc_path"])]
        self.batch_size = batch_size
        self.batch_idx = 0
        self.num_batches = len(self.samples_json) // batch_size

    def get_batch(self):
        if self.batch_idx % self.num_batches:
            self.shuffle_samples()

            # batch = random.sample(self.samples_json, self.batch_size)
        batch = self.samples_json[self.batch_size * self.batch_idx: self.batch_size * (self.batch_idx + 1)]
        mfccs_list = [np.load(sample["mfcc_path"]) for sample in batch]
        mfccs = self.padd_concat_mfccs(mfccs_list)
        labels = np.array([sample["label"] for sample in batch])
        self.batch_idx += 1
        return mfccs, labels

    def padd_concat_mfccs(self, mfccs_list):
        max_length = max([mfcc.shape[0] for mfcc in mfccs_list])
        mfccs_array = np.zeros((len(mfccs_list), max_length, mfccs_list[0].shape[-1]), dtype=np.float32)
        for idx, mfcc in enumerate(mfccs_list):
            mfcc = np.expand_dims(mfcc, 0)
            mfccs_array[idx, :mfcc.shape[1], :mfcc.shape[2]] = mfcc
        return mfccs_array

    def shuffle_samples(self):
        random.shuffle(self.samples_json)
        self.batch_idx = 0
