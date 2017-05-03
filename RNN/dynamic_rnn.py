#import tensorflow as tf
import random


class ToySequenceData(object):

    def __init__(self, n_samples=1000, max_seq_len=20, min_seq_len=3, max_value=1000):
        self.data = []
        self.labels = []
        self.seqlen = []
        for i in range(n_samples):
            len = random.randint(min_seq_len, max_seq_len)
            self.seqlen.append(len)

            if random.random() < .5:
                rand_start = random.randint(0, max_value - len)
                s = [[float(i) / max_value]
                     for i in range(rand_start, rand_start + len)]
                s += [[0.] for i in range(max_seq_len - len)]
                self.data.append(s)
                self.labels.append([1., 0.])
            else:
                s = [[float(random.randint(0, max_value)) / max_value]
                     for i in range(len)]
                s += [[0.] for i in range(max_seq_len - len)]
                self.data.append(s)
                self.labels.append([0., 1.])
        self.batch_id = 0

    def next(self, batch_size):
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(
            self.batch_id + batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(
            self.batch_id + batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(
            self.batch_id + batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen


learning_rate = 0.01
training_iters = 1000000
batch_size = 128
display_step = 10

seq_max_len = 20
n_hidden = 64
n_classes = 2

trainset = ToySequenceData(n_samples=1000, max_seq_len=seq_max_len)
testset = ToySequenceData(n_samples=500, max_seq_len=seq_max_len)
