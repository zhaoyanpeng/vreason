import math
import time
import torch
import warnings
from collections import defaultdict, namedtuple

import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

class ExpDecayLR(StepLR):
    def __init__(self, optimizer, step_size, gamma=0.5, last_epoch=-1, verbose=False, **kwargs):
        super().__init__(optimizer, step_size, gamma=gamma, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " +
                "please use `get_last_lr()`.", UserWarning
            )
        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        return [base_lr * self.gamma ** (self.last_epoch / self.step_size) for base_lr in self.base_lrs]

torch.optim.lr_scheduler.ExpDecayLR = ExpDecayLR # append a module

class SlotattnLR(StepLR):
    def __init__(self, optimizer, step_size, warmup_step=0, gamma=0.5, last_epoch=-1, verbose=False, **kwargs):
        self.warmup_step = warmup_step # will be needed in a function call of the base class
        super().__init__(optimizer, step_size, gamma=gamma, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " +
                "please use `get_last_lr()`.", UserWarning
            )
        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        factor = 1
        if self.last_epoch < self.warmup_step:
            factor = self.last_epoch / self.warmup_step
        return [base_lr * factor * self.gamma ** (self.last_epoch / self.step_size) for base_lr in self.base_lrs]

torch.optim.lr_scheduler.SlotattnLR = SlotattnLR # append a module

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
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu()
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

class AGI:
    def __init__(self):
        super().__init__()
    def __call__(self, x1, x2):
        """ x1 is the gold and x2 is the prediction
        """
        B, N, H, W = x1.shape

        x1 = x1.permute(0, 2, 3, 1).reshape(B, -1, N)
        x2 = x2.permute(0, 2, 3, 1).reshape(B, -1, N)

        x1_group_ids = x1.argmax(-1)
        x2_group_ids = x2.argmax(-1)

        x1_onehot = F.one_hot(x1_group_ids).float()
        x2_onehot = F.one_hot(x2_group_ids).float()

        npoint = x1_onehot.sum([1, 2])

        nij = torch.einsum("bji,bjk->bki", x2_onehot, x1_onehot)
        a = nij.sum(1)
        b = nij.sum(2)
        
        rindex = (nij * (nij - 1)).sum([1, 2])
        aindex = (a * (a - 1)).sum(1)
        bindex = (b * (b - 1)).sum(1)
        
        expected_rindex = aindex * bindex / (npoint * (npoint - 1))
        max_rindex = (aindex + bindex) / 2
        ari = (rindex - expected_rindex) / (max_rindex - expected_rindex)

        def _all_equal(x):
            return (x == x[..., :1]).all(dim=-1)

        both_single_cluster = _all_equal(x1_group_ids) & _all_equal(x2_group_ids)
        new_ari = torch.ones_like(ari).where(both_single_cluster, ari)
        return new_ari
