import math
import time
import torch
from collections import defaultdict, namedtuple

class Stats(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self._stats = defaultdict(float)

    @property
    def stats(self):
        return self._stats

    def __call__(self, **kwargs):
        for k, v in kwargs.items():
            self._stats[k] += v

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = self.count = 0

    def __call__(self, val, n=1):
        self.count += n
        self.sum += val * n

    @property
    def average(self):
        return self.sum / self.count

class Statistics(object):
    def __init__(self, xent=0, kl=0, n_words=0, n_correct=0):
        self._kl = kl
        self._xent = xent
        self._n_words = n_words
        self._n_correct = n_correct
        self._start_time = time.time()

    def reset(self):
        self._kl = 0
        self._xent = 0
        self._n_words = 0
        self._n_correct = 0
        self._start_time = time.time()

    def update(self, stat):
        self._xent += stat._xent
        self._kl += stat._kl
        self._n_words += stat._n_words
        self._n_correct += stat._n_correct

    def accuracy(self):
        return 100 * (self._n_correct / self._n_words)

    def xent(self):
        return self._xent / self._n_words

    def kl(self):
        return self._kl / self._n_words

    def ppl(self):
        return math.exp(min(self._xent / self._n_words, 100))

    def elbo(self):
        return math.exp(min((self._xent + self._kl) / self._n_words, 100))

    def elapsed_time(self):
        return time.time() - self._start_time

    def report(self):
        return (
            f"acc: {self.accuracy():.2f} elbo: {self.elbo():.2f} ppl: {self.ppl():.2f} " +
            f"xent: {self.xent():.2f} kl: {self.kl():.2f}"
        )

