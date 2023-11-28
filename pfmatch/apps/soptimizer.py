import os
from time import time

import torch
import yaml
from slar.optimizers import get_lr, optimizer_factory
from pfmatch.utils import CSVLogger, get_device, load_toymc_config
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from pfmatch.algorithms import PoissonMatchLoss, SirenTrack
from pfmatch.apps.toymc import ToyMCDataset


class SOptimizer:
    def __init__(self, cfg:dict):
        """Train the SirenVis model using track data.
        
        This class uses the ToyMCDataset to train the SirenVis model by using
        the model to predict the photo-electron (p.e.) spectrum from a track, and then comparing
        the predicted p.e. spectrum to the true p.e. spectrum using a Poisson loss function.
        
        Several classes are created using the configuration dictionary. Thus, you should include
        the corresponding keys for each class. The following classes are created:
        - `pfamtch.algorithms.SirenTrack`: a wrapper around `SirenVis` (cfg['model'])
        - `pfmatch.apps.toymc.ToyMCDataset`: a dataset for the track data (cfg['data']['dataset'])
            - if generating tracks, `photonlib.PhotonLib` is created (cfg['photonlib'])
            - if generating tracks, `pfmatch.apps.ToyMC` is created (cfg['ToyMC'] and cfg['Detector'])
        - `torch.io.data.DataLoader`: a dataloader for ToyMCDataset (cfg['data']['loader'])
        - `slar.utils.CSVLogger`: a logger for the training process (cfg['logger'])
        - `torch.optim.Optimizer`: an optimizer for the SirenVis model (cfg['train'])
        
        The main method is `train()`, which runs the training loop. See the example notebook
        `Train_SOptimizer.ipynb` for an example configuration and usage.
        """

        if cfg.get('device'):
            self._device = get_device(cfg['device']['type'])
        else:
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # essentials
        self._model = SirenTrack(cfg).to(self._device)
        self._criterion = PoissonMatchLoss().to(self._device)

        # training things
        dataset = ToyMCDataset(cfg)
        self._dataloader = {
            'train': DataLoader(dataset, collate_fn=dataset.collate_fn, **cfg['data']['loader']),
            'val': None
        }
        self._opt, epoch = optimizer_factory(self._model.parameters(), cfg)
        
        # resume training
        self.iteration, self.epoch = 0, 0
        if cfg['train'].get('resume') and cfg['model']['ckpt_file']:
            import re
            # file name goes as iteration-{}-epoch-{}.ckpt
            iteration, epoch = re.findall(r'iteration-(\d+)-epoch-(\d+).ckpt', cfg['model']['ckpt_file'])[0]
            self.iteration = int(iteration)
            self.epoch = int(epoch)
            print('[SOptimizer] Resuming training at iteration',iteration,'epoch',epoch)
        
        self._logger = CSVLogger(cfg)
        
        # learning rate scheduler
        lrs = cfg['train'].get('lr_scheduler',dict())
        self.scheduler = None
        if lrs.get('name'):
            if not hasattr(torch.optim.lr_scheduler,lrs['name']):
                raise RuntimeError(f'Learning rate scheduler not available: {lrs["Name"]}')
                
            val_split = lrs.get('validation_split', 0.1)
            n_val = int(len(dataset)*val_split)
            n_train = len(dataset) - n_val
            
            if isinstance(cfg['ToyMC'], str):
                seed = load_toymc_config(cfg['ToyMC'])['ToyMC'].get('NumpySeed', 0)
            else:
                seed = cfg.get('ToyMC', dict()).get('NumpySeed', 0)
            generator = torch.Generator().manual_seed(seed)
            train, val = random_split(dataset, [n_train, n_val], generator=generator)
            self._dataloader = {
                'train': DataLoader(train, collate_fn=dataset.collate_fn, **cfg['data']['loader']),
                'val': DataLoader(val, collate_fn=dataset.collate_fn, **cfg['data']['loader'])
            }   
            lrs_type = getattr(torch.optim.lr_scheduler,lrs['name'])
            lrs_args = lrs.get('parameters', dict())
            print('[lr_scheduler] Using learning rate scheduler',lrs['name'])
            print('[lr_scheduler] Using validation split',val_split,'for learning rate scheduler')
            self.scheduler = lrs_type(self._opt, **lrs_args)


        # training hyperparameters
        train_cfg = cfg.get('train',dict())
        self.epoch_max = train_cfg.get('max_epochs',int(1e20))
        self.iteration_max = train_cfg.get('max_iterations',int(1e20))
        self.save_every_iterations = train_cfg.get('save_every_iterations',-1)
        self.save_every_epochs = train_cfg.get('save_every_epochs',-1)

        # save config
        with open(os.path.join(self._logger.logdir,'train_cfg.yaml'), 'w') as f:
            yaml.safe_dump(cfg, f)


    @property
    def model(self):
        """SirenVis model"""
        return self._model
    
    @property
    def opt(self):
        """torch optimizer (Adam)"""""
        return self._opt
    
    @property
    def device(self):
        """Torch device being used"""
        return self._device
    
    @property
    def criterion(self):
        """Loss function for the training process (PoissonMatchLoss)"""
        return self._criterion
        
    @property
    def dataloader(self):
        """torch dataloader for the ToyMC dataset"""
        return self._dataloader
    
    @property
    def logger(self):
        """logger for the training process"""
        return self._logger


    def step(self, input):
        """Runs a single step of the optimizer.

        Parameters
        ----------
        input : dict
            a dictionary containing a batch of qpt_v, pe_v, and q_sizes, as returned by
            `ToyMCDataset.__getitem__`.
        """

        input['qpt_v'] = input['qpt_v'].to(self.device)
        input['pe_v'] = input['pe_v'].to(self.device)
        input['q_sizes'] = input['q_sizes'].to(self.device)
        input['weights'] = input['weights'].to(self.device)

        # run model
        out = self.model(input)
        target = input['pe_v']
        pred = out['pe_v']
        weights = input['weights']
        
        # compute loss
        loss = self.criterion(pred, target, weights)

        return target, pred, loss
    
    def train(self):
        """
        Trains SIREN using the ToyMC dataset.
        """
        twait = time()
        stop_training = False            
    
        # epoch loop
        while self.iteration < self.iteration_max and \
              self.epoch < self.epoch_max:  
                  

            if self.scheduler:
                # validate every epoch
                with torch.no_grad():
                    val_loss = 0
                    for batch in self.dataloader['val']:
                        val_loss += self.step(batch)[-1]
                    val_loss /= len(self.dataloader['val'])
                self.scheduler.step(val_loss)
                  
            # iteration loop (batch loop)
            for batch in tqdm(self.dataloader['train'], desc='Epoch %-3d'%self.epoch, unit='batch'):
                self.iteration += 1
                twait = time() - twait

                # step the model                    
                ttrain = time()
                target, pred, loss = self.step(batch)
                # backprop
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                ttrain = time() - ttrain
                
                
                # log training parameters
                cols = ['iter','epoch','loss','ttrain','twait', 'lr']
                vals = [self.iteration, self.epoch, loss.item(), ttrain, twait, get_lr(self.opt)]

                if self.scheduler:
                    cols.append('val_loss')
                    vals.append(val_loss)
                self.logger.record(cols, vals)

                twait = time()

                # step the logger (pe spectrum)
                self.logger.step(self.iteration, target, pred)
                
                # save model params periodically after iterations
                if self.save_every_iterations > 0 and \
                    self.iteration % self.save_every_iterations == 0:
                    self.save()
                    
                if self.iteration_max <= self.iteration:
                    stop_training = True
                    break


            if stop_training:
                break
                        
            self.epoch += 1
            
            # save model params periodically after epochs
            if (self.save_every_epochs*self.epoch) > 0 and self.epoch % self.save_every_epochs == 0:
                self.save(count=self.iteration/len(self.dataloader['train'].dataset))
            
        print('[SOptimizer] Stopped training at iteration',self.iteration,'epochs',self.epoch)
        self.logger.write()
        self.logger.close()


    def save(self, count=None):
        """Saves the model parameters to a checkpoint file."""
        if count is None:
            count = self.iteration/len(self.dataloader['train'].dataset)

        filename = os.path.join(self.logger.logdir,'iteration-%06d-epoch-%04d.ckpt')
        self.model.save_state(filename % (self.iteration, self.epoch), self.opt, count)
