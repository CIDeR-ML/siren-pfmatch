import numpy as np
import torch
from torch.utils.data import DataLoader

class SimpleLoader:
    def __init__(self, ds, batch_size, concat=False, drop_last=True):

        self.batch_size   = batch_size
        self.concat       = concat
        self.drop_last    = drop_last
        self._start_index = 0
        self._last_index  = self.batch_size

        self._ds = ds
        self._all_idx = np.arange(len(ds))

    def __len__(self):
        num_iter = int(len(self._ds) / self.batch_size)
        if not self.drop_last and len(self._ds) > (num_iter*self.batch_size):
            num_iter += 1
        return num_iter

    def __iter__(self):
        return self

    def __next__(self):

        if self._start_index >= len(self._ds):
            self._start_index = 0
            raise StopIteration

        self._last_index = self._start_index + self.batch_size

        if self._last_index <= len(self._ds):
            idx_v = self._all_idx[self._start_index:self._last_index]
            self._start_index = self._last_index

        elif self._drop_last:
            self._start_index = 0
            raise StopIteration

        else:
            idx_v = self._all_idx[self._start_index:]
            self._start_index = len(self._ds)

        res = self._ds.select(idx_v,self.concat)
        batch = dict(qclusters=res[0], flashes=res[1])
        if self.concat:
            batch['sizes']=res[2]
        return batch



class RandomLoader:
    def __init__(self, ds, batch_size, concat=False, drop_last=True):

        self.batch_size   = batch_size
        self.concat       = concat
        self.drop_last    = drop_last
        self._start_index = 0
        self._last_index  = self.batch_size

        self._ds = ds
        self._all_idx = np.arange(len(ds))
        np.random.shuffle(self._all_idx)

    def __len__(self):
        num_iter = int(len(self._ds) / self.batch_size)
        if not self.drop_last and len(self._ds) > (num_iter*self.batch_size):
            num_iter += 1
        return num_iter

    def __iter__(self):
        return self

    def __next__(self):

        if self._start_index >= len(self._ds):
            self._start_index = 0
            np.random.shuffle(self._all_idx)
            raise StopIteration

        self._last_index = self._start_index + self.batch_size

        if self._last_index <= len(self._ds):
            idx_v = self._all_idx[self._start_index:self._last_index]
            self._start_index = self._last_index

        elif self.drop_last:
            self._start_index = 0
            np.random.shuffle(self._all_idx)
            raise StopIteration

        else:
            idx_v = self._all_idx[self._start_index:]
            self._start_index = len(self._ds)

        res = self._ds.select(idx_v,self.concat)
        batch = dict(qclusters=res[0], flashes=res[1])
        if self.concat:
            batch['sizes']=res[2]
        return batch



class RandomSequenceLoader:


    def __init__(self, ds, batch_size, concat=False, drop_last=True):

        self._iter_index  = 0
        self._iter_max    = int(len(ds)/batch_size)
        self.batch_size   = batch_size
        self.concat       = concat
        self.drop_last    = drop_last

        if (len(ds)%self.batch_size)<1 and not self.drop_last:
            self.drop_last = True

        self._ds = ds


    def __len__(self):
        return self._iter_max if self.drop_last else self._iter_max+1

    def __iter__(self):
        return self

    def __next__(self):

        num_samples = self.batch_size

        if self._iter_index > self._iter_max:
            self._iter_index = 0
            raise StopIteration

        if self._iter_index == self._iter_max:

            if self.drop_last:
                self._iter_index = 0
                raise StopIteration

            else:
                num_samples = len(ds) % self.batch_size


        start_index = int(np.random.uniform()*(len(self._ds)-num_samples))

        res = self._ds.slice(start_index,start_index+num_samples)
        
        batch = dict(qclusters=res[0], flashes=res[1], sizes=res[2])
        if not self.concat:
            batch['qclusters']=torch.split(batch['qclusters'],batch['sizes'])
            batch.pop('sizes')          

        self._iter_index += 1
        return batch




