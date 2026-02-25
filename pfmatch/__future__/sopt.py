import torch
from time import time

from tqdm.auto import tqdm
from slar.nets import SirenVis, MultiVis
from slar.optimizers import get_lr
from pfmatch.utils import scheduler_factory, CSVLogger
from pfmatch.algorithms import PoissonMatchLoss
from pfmatch.__future__.io_factories import loader_factory
from pfmatch.utils import import_from

def criterion_factory(cfg):
    crit_cfg = cfg['train'].get('criterion', {}).copy()
    class_name = crit_cfg.pop('class', None)
    if class_name is None:
        return PoissonMatchLoss()

    CritCls = import_from(class_name)
    crit = CritCls(**crit_cfg)
    return crit

class SOptimizer:
    def __init__(self, cfg, device=None, resume=True):
        if 'multivis' in cfg:
            print('[SOptimizer] Using MultiVis')
            model = MultiVis(cfg)
        elif 'model' in cfg:
            print('[SOptimizer] Using SirenVis')
            model = SirenVis(cfg)
        else:
            raise NotImplementedError('No "multivis" or "model" in cfg')
        
        self._model = model.to(device)
        
        train_cfg = cfg.get('train')
        self._optimizer = torch.optim.Adam(
            self._model.parameters(), **train_cfg.get('optimizer', {})
        )

        self.lr0 = self.lr
        print('[SOptimizer] lr0:', self.lr)
        
        self._scheduler = scheduler_factory(
            self._optimizer, train_cfg.get('scheduler', {}), verbose=True,
        )
        
        self._dataloader = loader_factory(cfg)
        self._logger = CSVLogger(cfg)
        
        self.criterion = criterion_factory(cfg)
        print('[Criterion]', self.criterion)

        self.epoch = 0
        self.current_loss = 1.e9
        self.n_iter = 0
        self.epoch_max = train_cfg.get('max_epochs', int(1e20))
        self.iter_max = train_cfg.get('max_iterations', int(1e20))
        self.save_every_iters = train_cfg.get('save_every_iterations', -1)
        self.save_every_epochs = train_cfg.get('save_every_epochs', -1)
    
    
    @property
    def device(self):
        return self._model.device
    
    @property
    def lr(self):
        return get_lr(self._optimizer)

    def step(self, batch):
        device = self.device
        
        # prepare input to device
        if isinstance(batch['qclusters'],list):
            qpts = torch.cat(batch['qclusters']).to(device)
            sizes = list(map(len, batch['qclusters']))
        else:
            qpts  = batch['qclusters']
            sizes = batch['sizes']
        if isinstance(batch['flashes'],list):
            target = torch.stack(batch['flashes']).to(device)
        else:
            target = batch['flashes']

        #tstart=time()
        vis_q = self._model.visibility(qpts[:,:3]) * qpts[:,-1].unsqueeze(-1)    
        pred = torch.stack([arr.sum(axis=0) for arr in torch.split(vis_q, sizes)])
        loss = self.criterion(pred, target)
        return loss
    
    def train(self):
        while self.epoch < self.epoch_max and self.n_iter < self.iter_max:
            self.train_one_epoch()
        self._logger.write()
        self._logger.close()
    
    def train_one_epoch(self):
        
        t_start = time()
        for batch in tqdm(self._dataloader, desc=f'epoch {self.epoch}, loss {self.current_loss: .2f}'):
            t_wait = time() - t_start
            
            t_start = time()
            self._optimizer.zero_grad()
            loss_pred = self.step(batch)
            t_forward = time() - t_start
            
            loss = loss_pred.mean()
            loss.backward()
            self._optimizer.step()
            #TODO(2024-03-27 kvt) add step(loss)
            if self._scheduler is not None:
                self._scheduler.step()
            
            t_train = time() - t_start
            self.current_loss = loss.item()

            # Logger
            log_dict = {
                'iter': self.n_iter,
                'epoch': self.epoch,
                'loss': self.current_loss,
                'twait': t_wait,
                'ttrain' : t_train,
                'tforward' : t_forward,
                'lr': self.lr/self.lr0,
            }
            self.log(log_dict)
            self._logger.step(self.n_iter)
            
            if (self.save_every_iters > 0
                and self.n_iter % self.save_every_iters == 0):
                self.save()
            
            self.n_iter += 1
            if self.n_iter >= self.iter_max:
                break
            t_start = time()
            
        if (self.save_every_epochs > 0 
                and self.epoch % self.save_every_epochs == 0):
                self.save()
                
        self.epoch += 1
        
    def log(self, log_dict):
        cols = list(log_dict.keys())
        values = list(log_dict.values())
        self._logger.record(cols, values)
        
    def save(self):
        import os
        fpath = os.path.join(
            self._logger.logdir,
            f'iteration-{self.n_iter:08}-epoch-{self.epoch:05}.ckpt'
        )
        self._model.save_state(fpath, self._optimizer)
        
    def load(self):
        pass
