import os
import torch
import numpy as np
DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch.utils.data import Dataset
from contextlib import closing
from pfmatch.io import H5File

class SirenCalibDataset(Dataset):
    def __init__(self, filepath, n_pmts):
        super().__init__()
        
        if isinstance(filepath, str):
            filepath = [filepath]
        
        files = filepath
        self._build_file_index(files)
        self._n_pmts = n_pmts
    
    def _build_file_index(self, files):
        cnts = torch.zeros(len(files), dtype=int)
        
        self._files = []
        for i, fpath in enumerate(files):
            if not os.path.isfile(fpath):
                raise FileNotFoundError(fpath)
                
            #with closing(H5File.open(fpath, 'r', verbose=False)) as f:
            #    cnts[i] = len(f)
            f = H5File.open(fpath, 'r', verbose=False)
            self._files.append(f)
            cnts[i] = len(f)
        
        #self._files = files
        self._file_idxs = torch.cat([torch.tensor([0]), cnts.cumsum(0)])
                   
    def __del__(self):
        while len(self._files) > 0:
            f = self._files.pop()
            f.close()

    def __len__(self):
        return self._file_idxs[-1]

    def __getitem__(self, idx):
        return idx

    def read_many(self, idx):
        idx = torch.as_tensor(idx)
        file_idxs = torch.bucketize(idx, self._file_idxs, right=True) - 1
        
        key_q = 'qpt_v'
        key_f = 'pe_v'
        
        output = {
            'qclusters' : [],
            'flashes' : [],
            'ev_idxs' : [],
            'file_idxs' : [],
        }
        
        for fid in file_idxs.unique():
            mask = file_idxs==fid
            offset = self._file_idxs[fid]
            ev_idxs = (idx[mask] - offset).tolist()
            ev_idxs.sort()
        
            #fpath = self._files[fid]
            #with closing(H5File.open(fpath, 'r', verbose=False)) as fin:
            #    q_vv, f_vv = fin.read_many(ev_idxs, self._n_pmts)
            #del fin
            fin = self._files[fid]
            q_vv, f_vv, _ = fin.read_many(ev_idxs, self._n_pmts)
               
            # fill qclusters
            for q_v in q_vv:
                # correct for the true position
                qpts = getattr(q_v[0], key_q)
                qpts[:,0] += q_v[0].xmin_true - qpts[:,0].min()
                output['qclusters'].append(qpts)
                #output['qclusters'].append(getattr(q_v[0], key_q))
            
            # fill flashes
            for f_v in f_vv:
                output['flashes'].append(getattr(f_v[0], key_f))
         
            # fill file and event indices
            output['ev_idxs'].append(ev_idxs)
            output['file_idxs'].append(fid)
            
        return output


class TracksInMemory(Dataset):
    def __init__(self, filepath, n_pmts, 
                 device=DEFAULT_DEVICE, 
                 flash_key='pe_v'):

        super().__init__()
        self._n_pmts = n_pmts
        self.device = device
        self.qcluster_v = []
        self.flash_v = []
        self._key_f = flash_key
        self._read_files(filepath)
    
    def device(self):
        return self.device
    
    def _read_files(self,files):

        self.qcluster_v = []
        self.flash_v = []
        for i,fname in enumerate(files):
            f = H5File.open(fname,'r')
            print(f'[TracksInMemory] reading file {i}/{len(files)} from {fname} with {len(f)} entries')
            qs,fs,_ = f.read_many(np.arange(len(f)),self._n_pmts,verbose=True)
            print(f'[TracksInMemory] combining entries into single list...')
            qs = np.concatenate(qs)
            fs = np.concatenate(fs)
            qs = [torch.as_tensor(qc.qpt_v,device=self.device) for qc in qs]
            #fs = [torch.as_tensor(f.pe_v[None,:],device=self.device) for f in fs]
            fs = [
                torch.as_tensor(getattr(f, self._key_f)[None,:],
                                device=self.device)
                for f in fs 
            ]
            assert len(qs)==len(fs), 'Number of QCluster and Flash do not match'
            self.qcluster_v += qs
            self.flash_v += fs
            f.close()        
        
    def __len__(self):
        return len(self.qcluster_v)
    
    def __getitem__(self,idx):
        return self.qcluster_v[idx], self.flash_v[idx]
    
    def select(self,idx_v,concat=False):
        
        qs=[self.qcluster_v[idx] for idx in idx_v]
        fs=[self.flash_v[idx] for idx in idx_v]
        
        if not concat:
            return qs, fs
        
        ls=torch.as_tensor([len(q) for q in qs])
        qs=torch.concat(qs)
        fs=torch.concat(fs)
        return qs, fs, ls

    
class TracksInConsecutiveMemory(Dataset):
    def __init__(self, filepath, n_pmts, 
                 device=DEFAULT_DEVICE,
                 flash_key='pe_v'):
        super().__init__()
        self._n_pmts = n_pmts
        self.device = device
        self.qcluster_v = []
        self.flash_v = []
        self._key_f = flash_key
        self._read_files(filepath)
    
    def device(self):
        return self.device
    
    def _read_files(self,files):

        self.qcluster_v = []
        self.flash_v = []
        for i,fname in enumerate(files):
            f = H5File.open(fname,'r')
            print(f'[TracksInMemory] reading file {i}/{len(files)} from {fname} with {len(f)} entries')
            qs,fs,_ = f.read_many(np.arange(len(f)),self._n_pmts,verbose=True)
            print(f'[TracksInMemory] combining entries into single list...')
            qs = np.concatenate(qs)
            fs = np.concatenate(fs)
            qs = [torch.as_tensor(qc.qpt_v,device=self.device) for qc in qs]
            #fs = [torch.as_tensor(f.pe_v[None,:],device=self.device) for f in fs]
            fs = [
                torch.as_tensor(getattr(f, self._key_f)[None,:],
                                device=self.device)
                for f in fs 
            ]
            assert len(qs)==len(fs), 'Number of QCluster and Flash do not match'
            self.qcluster_v += qs
            self.flash_v += fs
            f.close()
        
        print(f'[TracksInMemory] creating single consecutive tensor...')
        self.len_v = tuple([len(qc) for qc in self.qcluster_v])
        self.qcluster_v = torch.concat(self.qcluster_v).to(self.device)
        self.flash_v = torch.concat(self.flash_v).to(self.device)

    def __len__(self):
        return len(self.len_v)
    
    def __getitem__(self,idx):
        
        start_idx = sum(self.len_v[:idx])
        end_idx   = sum(self.len_v[:(idx+1)])
        return self.qcluster_v[start_idx:end_idx], self.flash_v[idx*self._n_pmts:(idx+1)*self._n_pmts]
    
    def select(self, idx_v, concat=False):
        
        qs = [self.qcluster_v[sum(self.len_v[:idx]):sum(self.len_v[:(idx+1)])] for idx in idx_v]
        fs = [self.flash_v[idx*self._n_pmts:(idx+1)*self._n_pmts] for idx in idx_v]
        if not concat:
            return qs, fs
        qs = torch.concat(qs)
        fs = torch.concat(fs)
        ls = tuple([self.len_v[idx] for idx in idx_v])
        
        return qs,fs,ls
    
    def slice(self, start, end):
        start_idx = sum(self.len_v[:start])
        end_idx   = sum(self.len_v[:end])
        qs = self.qcluster_v[start_idx:end_idx]
        fs = self.flash_v[start:end]
        ls = self.len_v[start:end]
        
        return qs, fs, ls
        

