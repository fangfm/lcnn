# -*- coding: utf-8 -*-

import numpy as np
import os
from sklearn.utils import shuffle
import sys
import struct

class dataio(object):
    def __init__(self, train_genuine, train_spoof, dev_genuine, dev_spoof,
                 test_data=None, batch_size=32):
        
        if train_genuine is not None and train_spoof is not None:
            self.training_data, self.label = self.load_data(train_genuine, train_spoof)
            self.frames = len(self.training_data)
            self.batch_size = min(batch_size, self.frames)
            self.max_index = self.frames - self.batch_size
        
        if dev_genuine is not None and dev_spoof is not None:
            self.dev_data, self.dev_label = self.load_data(dev_genuine, dev_spoof)
            self.dev_frames = len(self.dev_data)
            self.current_dev_index = 0
            self.dev_batch_size = min(64, self.dev_frames)
            self.dev_iterations = (self.dev_frames - 1)//self.dev_batch_size + 1

        if test_data is not None:
            self.test_data, self.test_names = self.load_test_data(test_data, 400)
            self.test_frames = len(self.test_data)

    def load_data(self, scp_genuine, scp_spoof):
        if scp_genuine is not None:
            genuine = self._load_data(scp_genuine, 400)
            genuine = np.reshape(genuine, (-1, 864, 400, 1))
            genuine_lab = np.zeros((len(genuine), 2), dtype=np.float32)
            genuine_lab[:, 0] = 1.0
            
        if scp_spoof is not None:
            spoof = self._load_data(scp_spoof, 400)
            spoof = np.reshape(spoof, (-1, 864, 400, 1))
            spoof_lab = np.zeros((len(spoof), 2), dtype=np.float32)
            spoof_lab[:, 1] = 1.0
            
        if scp_genuine is not None and scp_spoof is not None:
            x = np.concatenate((genuine, spoof), axis=0)
            y = np.concatenate((genuine_lab, spoof_lab), axis=0)
        elif scp_genuine is not None and scp_spoof is None:
            x = genuine
            y = genuine_lab
        elif scp_genuine is None and scp_spoof is not None:
            x = spoof
            y = spoof_lab
        else:
            raise NotImplementedError
            
        return x, y

    def _load_data(self, scp_path, dim):
        scp = np.loadtxt(scp_path, dtype=str)
        
        total_frames = 0
        for name in scp:
            total_frames += os.path.getsize(name)/4/dim
            
        data = np.zeros((total_frames, dim), dtype=np.float32)
            
        idx = 0
        for name in scp:
            with open(name, 'rb') as f:
                v = f.read()
                v = np.frombuffer(v, dtype=np.float32)
                v = np.reshape(v, (-1, dim))
                data[idx:idx+len(v)] = v
                
                idx += len(v)

        return data

    def load_test_data(self, scp_path, dim):
        scp = np.loadtxt(scp_path, dtype=str)

        test_data = list()
        for name in scp:
            with open(name, 'rb') as f:
                v = f.read()
                v = np.frombuffer(v, dtype=np.float32)
                v = np.reshape(v, (-1, 864, dim, 1))
                test_data.append(v)

        return test_data, scp

    def shuffle(self):
        self.training_data, self.label = shuffle(self.training_data, self.label)

    def batch(self):
        rand_v = np.random.randint(self.max_index)

        x = self.training_data[rand_v:rand_v+self.batch_size]
        y = self.label[rand_v:rand_v+self.batch_size]

        return x, y

    def dev_batch(self):
        s = self.current_dev_index
        e = s + self.dev_batch_size

        if e > self.dev_frames:
            e = self.dev_frames

        x = self.dev_data[s:e]
        y = self.dev_label[s:e]

        if e >= self.dev_frames:
            self.current_dev_index = 0
        else:
            self.current_dev_index = e
            
        return x, y

    def save_data(self, name, data):
        with open(name,'wb') as f:
            f.write(struct.pack('f'*data.size, *data.flat))
