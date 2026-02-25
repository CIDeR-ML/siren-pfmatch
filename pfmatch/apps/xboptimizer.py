from __future__ import annotations

import time
import numpy as np
import torch

from itertools import compress
from tqdm.auto import trange
from photonlib import PhotonLib, MultiLib
from slar.nets import SirenVis, MultiVis
from pfmatch.datatypes import QCluster, Flash, FlashMatchInput
from pfmatch.algorithms import  MultiFlashHypothesis
from pfmatch.utils import scheduler_factory
from scipy.optimize import linear_sum_assignment

def criterion_factory(cfg={}):
    """
    A factory to create loss function from cfg dict.
    Default: PoissonMatchLoss

    Example:
    ```
    cfg = {
        'Class': 'Chi2Loss', 
        'Param': {
            'eps': 0.001}
        }
    }

    crit = criterion_factory(cfg)
    ```

    """
    import pfmatch.algorithms as module
            
    class_name = cfg.get('Class', 'PoissonMatchLoss')
    kwargs = cfg.get('Param', {})
            
    return getattr(module, class_name)(**kwargs)

class XBatchOptimizer:
    '''
    Optimizer of x-offset in batch for a list of charge-flash pairs.
    Optianal mode for flash matching (finding the best permutated pair).

    Arguments
    ---------
    cfg: dict
        Dictionary for the optimizer parameters.
        Defaut values will be taken if key name `XBatchOptimizer` is not found.

    vis_model: PhotonLib | SirenVis
        Visibility model in form of PhotonLib (aka LUT) or SirenVis.

    device: torch.device, optional
        The device to the contructed optimizer. If None, then the device is on
        CPU by default.

    Configuration
    -------------
    Below is an example configuration (with default values) in yaml foramt.

    ```
    XBatchOptimizer:
        # -------------------------------------------------------
        # Input Data Key (Optinal)
        # Default: {'qclsuter':'qcluster_v', 'flash':'pe_v'}
        # Alternatives are 'raw_qcluster_v', 'pe_true_v'
        # -------------------------------------------------------
        # DataKey: 
        #    qcluster: qcluster_v
        #    flash: pe_v 
        
        # -------------------------------------------------------
        # Optimization parameters
        # -------------------------------------------------------
        InitLearningRate: 1.0
        MaxIteration: 500

        # -------------------------------------------------------
        # Conditions for early stop
        # -------------------------------------------------------
        StopPatience: 20
        StopDeltaXMin: 0.1
        StopDeltaLoss: 0.001
        
        # -------------------------------------------------------
        # Learning Rate Scheduler (Optional)
        # -------------------------------------------------------
        # Scheduler:
        #    Class: PolynomialLR
        #    Param:
        #        total_iters: 200

        # -------------------------------------------------------
        # Do matching.
        # If `False`, assume charge-flash are already matched.
        # -------------------------------------------------------
        Match: True 

        # -------------------------------------------------------
        # Perform loss scan with `LossScanStepSize` to determine 
        # initial x-offset. `Prefit` is set to `True` if both 
        # `Match` and `Prefilter` are `True`.
        # -------------------------------------------------------
        Prefit: True
        LossScanStepSize: 1.

        # -------------------------------------------------------
        # Prefilter pairs for matching only.
        # Not use if `Match: False`.
        # Keep pairs with loss < `PrefilterLoss` 
        # or the best `PrefilterTopK`-th pairs.
        # It is possible to run prefit without prefilter, which 
        # keeps all possible pairs w/ initial offset estimation.
        # -------------------------------------------------------
        Prefilter: True
        PrefilterTopK: 2
        PrefilterLoss: 200

        # -------------------------------------------------------
        # Loss function. Default: PoissonMatchLoss if not set
        # -------------------------------------------------------
        # Criterion:
        #    Class: Chi2Loss
        #    Param:
        #       eps: 0.001

        # -------------------------------------------------------
        # More messages?
        # -------------------------------------------------------
        Verbose: False
    ```
    '''
    def __init__(self, cfg:dict, vis_model: PhotonLib | MultiVis| SirenVis | MultiVis, device=None):
        self._vis_model = vis_model.to(device)

        this_cfg = cfg.get('XBatchOptimizer', dict())
        self.cfg = this_cfg

        self.initial_lr = this_cfg.get('InitLearningRate', 1.0)
        self.loss_scan_step = this_cfg.get('LossScanStepSize', 1.)
        self.max_iterations = this_cfg.get('MaxIteration', 500)
        self.stop_patience = this_cfg.get('StopPatience', 20)
        self.stop_xmin_delta = this_cfg.get('StopDeltaXMin', 0.1)
        self.stop_loss_delta = this_cfg.get('StopDeltaLoss', 0.001)
        self.prefilter_topk = this_cfg.get('PrefilterTopK', 2)
        self.prefilter_loss = this_cfg.get('PrefilterLoss', 200)
        self.do_prefit = this_cfg.get('Prefit', True)
        self.do_prefilter = this_cfg.get('Prefilter', True)
        self.do_match = this_cfg.get('Match', False)
        self.verbose = this_cfg.get('Verbose', False)

        # For mathcing w/ prefilter, do prefit by default
        if self.do_match and self.do_prefilter:
            self.do_prefit = True

        # default data keys
        self.data_key = {'qcluster':'qcluster_v', 'flash':'pe_v'}
        self.data_key.update(this_cfg.get('DataKey', {}))

        self.crit = criterion_factory(this_cfg.get('Criterion', {}))

        # verbose messages
        if len(this_cfg) == 0:
            self.print('[XBatchOptimizer] use default configuration')

        self.print(f'[XBatchOptimizer] {type(vis_model)}, device={self.device}')
        self.print('[XBatchOptimizer]', self.crit)

        if ((isinstance(vis_model, SirenVis) or isinstance(vis_model, MultiVis))
                and self.device==torch.device('cpu')):
            self.print('[XBatchOptimizer] SirenVis/MultiVis on CPU could be SLOW !!!')

    def print(self, *values, **kwargs):
        if self.verbose:
            print(*values, **kwargs)

    @property
    def device(self):
        return self._vis_model.device

    @property
    def x_ranges(self):
        return self._vis_model.meta.x

    @staticmethod
    def pair_product(m : int, n : int, device):
        '''
        Make all index combinations of `m * n` pairs.

        Arguments
        ---------
        m, n : int, int
            Number of elements.

        device: torch.device
            Output device

        Returns
        -------
        pairs: tuple(tensor, tensor)
            Indices of all pair combinations.
        '''
        pairs = torch.cartesian_prod(
            torch.arange(m, device=device),
            torch.arange(n, device=device)
        )
        pairs = pairs[:,0], pairs[:,1]
        return pairs

    def get_dx_steps(self, qpt_v, step_size=None):
        '''
        Generate steps from a charge cluster. The ranges are calculated from the
        minimum distances between the charge cluster and the x-boundaries of
        the active volume from `meta` instance.

        Arguments
        ---------
        qpt_v: torch.Tensor
            2-dim tensor of the charge cluster.
            `qpt_v[:,0]` are the coordinates along the drift direction.

        step_size: int, optinal
            Step size, default `None`. If not set, use ``self.loss_scan_step``.

        Returns
        -------
        dx: torch.Tensor
            1-dim vector of offset in x. Same output device as the optimizer.
        '''

        x_min, x_max = self.x_ranges
        x = qpt_v[:,0]

        dx_min, dx_max = x_min - x.min(), x_max - x.max()
        dx = torch.arange(
            dx_min, dx_max, self.loss_scan_step, device=self.device
        )
        return dx

    def scan_loss(self, qpt_v, tgt_pe):
        '''
        Perform loss scan of a charge cluster with targets of optical flash(es).

        Arguments
        ---------
        qpt_v: torch.Tensor
            Charge cluster with shape `(N,4)`.
            `qpt_v[:,:3]` - (x,y,z) coordinates
            `qpt_v[:,3]` - charge values

        tgt_pe: torch.Tensor
            Optical flash data with shape `(N_pmt,)` or
            `(N_flash,N_pmt)`.

        Returns
        -------
        dx: torch.Tensor
            Steps in the drift direction with shape `(N_step,)`.
            It is generated by ``self.get_dx_step()``.

        loss: torch.Tensor
            Loss values evaluated at each dx offset step.
            For a single flash input, the shape is `(N_step,)`.
            For multiple flashes input, the shape is `(N_step,N_flash)`.
        '''

        dx = self.get_dx_steps(qpt_v)
        n_steps = len(dx)
        sizes = torch.full((n_steps,), len(qpt_v), device=self.device)
        batch = qpt_v.repeat(n_steps, 1)

        with torch.no_grad():
            out_pe = MultiFlashHypothesis.apply(
                dx, batch, sizes, self._vis_model
            )

        if tgt_pe.ndim == 1: 
            loss = self.crit(out_pe, tgt_pe)
        elif tgt_pe.ndim == 2:
            loss = self.crit(out_pe.unsqueeze(0), tgt_pe.unsqueeze(1))
        else:
            raise RuntimeError(
                f'Dimension of tgt_pe ({tgt_pe.ndim}) must be 1 or 2.'
            )

        return dx, loss

    def collate_input(self, input: FlashMatchInput):
        '''
        Collation of input data.
        See ``FlashMatchInput.collate_q` and ``FlashMatchInput.collate_f``.

        Arguments
        ---------
        input : FlashMatchInput
            An instance of ``FlashMatchInput``.

        Returns
        -------
        data : dict
            Collated data in a dictonary.

            'qclusters' : list(tensor)
                List of position and charge point for all clusters.

            'flashes' : tensor
                Flashes (p.e.) as a 2-dim tensor (n_flashes, n_pmts)

            'dx_ranges' : tensor
                Ranges of x-offset for all clusters.,
        '''
        key_q = self.data_key['qcluster']
        key_f = self.data_key['flash']
        device = self.device

        self.print(f'[collate_input] data keys ({key_q}, {key_f})')
        data = dict(
            qclusters=input.collate_q(device=device, key=key_q),
            flashes=input.collate_f(device=device, key=key_f),
            dx_ranges=input.dx_ranges(*self.x_ranges).to(device),
        )
        return data

    def make_input_batch(self, input : FlashMatchInput | dict, pairs=None):
        '''
        Prepare batch input for ``_fit()`` function. 

        Arguments
        ---------
        input: FlashMatchInput | dict
            An instance of `FlashMatchInput` or dictonary of collated input
            See ``collate_input()``.

        pairs: tuple(tensor), optional
            List of indices `(idx_q, idx_f)` for charge-flash pairs.
            Default: None.
            The lenghts `len(idx_q) == len(idx_f)` give the number of
            pairs.  See ``prefilter()`` on how to produce such pairs.

        Returns
        -------
        batch: dict
            Batch input data as a dictionary.

            `pairs`: tuple(tensor) or None
                Taken from function argument.

            `n_qcluster`, `n_flash`: int, int
                Number of charge clusters (flashes) in the `FlashMatchInput`
                instance. It may be different than the _batched_ qclsuters and
                flashes. See below.

            `qclusters`: Tensor
                Combined `qpt_v` of all charge clusters in a single tensor.
                A cluster is repeated if it appears multiple time in the
                `pairs`.

            `sizes`: Tensor
                Number of points per cluster. 

            `flashes`: Tensor
                Optical flashes in a single tensor. 
                Flash data is repeated the same way as `qclusters` above.  Each
                flash is a potential matching candidate to the corresponding
                qclusters in _batch_, `len(flashes) == len(sizes)`.

            `dx_ranges`: Tensor
                Ranges of x-offset for each `qclusters` in _batch_.
        '''

        device = self.device

        if isinstance(input, FlashMatchInput):
            input = self.collate_input(input)
        
        qclusters, flashes, dx_ranges = \
            tuple(map(input.__getitem__, ['qclusters','flashes','dx_ranges']))

        batch = {
            'pairs': pairs,
            'n_qcluster': len(qclusters),
            'n_flash': len(flashes),
        }

        # number of points per qcluster
        sizes = torch.tensor(
            list(map(len,qclusters)), dtype=int, device=device
        )

        if pairs is None:
            batch['flashes'] = flashes.to(device)
            batch['qclusters'] = torch.cat(qclusters).to(device)
            batch['sizes'] = sizes
            repeats = torch.ones(len(qclusters), dtype=int, device=device)
        else:
            sel_q, sel_f = pairs
            batch['flashes'] = flashes[sel_f]

            repeats = torch.zeros(len(qclusters), dtype=int, device=device)

            batch_q = []
            for i_q in sel_q:
                repeats[i_q] += 1
                batch_q.append(qclusters[i_q])
            batch['qclusters'] = torch.cat(batch_q)
            batch['sizes'] = sizes.repeat_interleave(repeats)

        batch['dx_ranges'] = dx_ranges.repeat_interleave(repeats, dim=0)
        return batch

    def prefit(self, qclusters, flashes):
        '''

        Perform loss scan for pairwise charge and flash input.  To scan all
        combination of charge and flash pair, use ``prefilter()`` in matching
        mode. Typical inputs are from the returns of ``make_input_tensors()``.

        Arguments
        ---------
        qclusters: list of Tensor
            A list of charge cluster points.

        flashes: Tensor or list of Tensor
            Corresponding optical flashes to qclusters in orders. The format
            can either a 2-dim tensor or a list of 1-d tensor.
            Prerequest: `len(qclusters) == len(flashes)`.

        Returns
        -------
        output: dict
            Dictonary of prefit output. `(key, value)` are described below.

            `dx_init`: Tensor
                Initial guess of dx offets from loss scan.
            `tspent`: float
                Time spent in seconds on this function call.
        ''' 

        t_start = time.time()

        self.print(f'[prefit] scan loss for {len(qclusters)} clusters, '
                   f'{len(flashes)} flashes')

        #TODO(2024-02-26 kvt) check len()
    
        dx_init = []
        for qpt_v, tgt_pe in zip(qclusters, flashes):
            dx, loss = self.scan_loss(qpt_v, tgt_pe)
        
            step_idx = loss.argmin()
            dx_init.append(dx[step_idx].item())
        
        t_stop = time.time()

        output = {
            'dx_init': torch.tensor(dx_init),
            'tspent': t_stop - t_start,
        }

        return output

    def select_top_k(self, loss_matrix):
        '''
        Select top-k pairs from loss matrix.

        Arguments
        ---------
        loss_matrix: tensor, (_N_,_M_)
            Loss matrix of _N_ clusters and _M_ flashes.

        Returns
        -------
        pairs: tuple(tensor, tensor)
            List of indices of cluster-flash pairs.
        '''

        threshold = self.prefilter_loss
        top_k = self.prefilter_topk
        self.print(f'[select_top_k] threshold={threshold}, top_k={top_k}')

        # select pairs below prefilter threshold 
        sel = loss_matrix < threshold

        # select top-k if no pair below threshold
        topk_q = torch.where(sel.count_nonzero(axis=1)==0)[0]
        topk_f = torch.where(sel.count_nonzero(axis=0)==0)[0]

        for i_q in topk_q:
            i_f = loss_matrix[i_q].argsort()[:top_k]
            sel[i_q, i_f] = True

        for i_f in topk_f:
            i_q = loss_matrix[:,i_f].argsort()[:top_k]
            sel[i_q, i_f] = True

        # make pairs
        pairs = torch.where(sel)
        return pairs

    def prefilter(self, qclusters, flashes):
        '''

        Perform loss scan on _all_ combinations of charge-flash pairs. Keep
        only pairs with loss below a threshold, or top _k_ pairs of smaller
        losses. See ``prefit()`` for the input descriptions.

        Arguments
        ---------
        qclusters: list of Tensor
            A list of charge cluster points.

        flashes: Tensor or list of Tensor
            A list of optical flashes

        Returns
        -------
        output: dict
            Dictonary of prefilter output. `(key, value)` are described below.

            `tspent`: float
                Time spent in seconds on this function call.
            `pairs`: list of (int, int)
                List of indices of `(i,j)`, where i-th charge cluster and j-th
                flash is a potental matching pair.
            `dx_init`: Tensor
                Initial guess of dx offets from loss scan for the potential pairs.
                Same lenght as `pairs`.
            '''
        t_start = time.time()

        self.print(f'[prefilter] scan loss for {len(qclusters)} clusters, '
                   f'{len(flashes)} flashes')

        loss_matrix = torch.empty(
            (len(qclusters), len(flashes)), device=self.device
        )
        dx_matrix = torch.empty_like(loss_matrix)

        for i_q, qpt_v in enumerate(qclusters):
            dx_step, loss_step = self.scan_loss(qpt_v, flashes)

            loss_min, step_idx = loss_step.min(axis=1)

            loss_matrix[i_q] = loss_min
            dx_matrix[i_q] = dx_step[step_idx]

        if not self.do_prefilter:
            self.print('[prefilter] '
                       'set "Prefilter: True" for faster running time')
            n_q, n_f = loss_matrix.shape
            device = loss_matrix.device
            pairs = self.pair_product(n_q, n_f, device)
        else:
            pairs = self.select_top_k(loss_matrix)
        dx_init = dx_matrix[pairs]

        self.print(
            f'[prefilter] select {len(dx_init)}/{loss_matrix.numel()} pairs'
        )
        t_stop = time.time()

        output = {
            'pairs': (pairs[0].cpu(), pairs[1].cpu()),
            'dx_init': dx_init.cpu(),
            'loss_matrix': loss_matrix.cpu(),
            'tspent': t_stop - t_start,
        }
        return output

    def match(self, output, batch):
        '''
        Perform bipartite match charge-flash pairs to minimize the total loss.

        Arguments
        ---------
        output: dict
            Output dictionary from ``fit()``, which gives `loss_best` and
            `dx_best` for the charge-flash pair candidates.

        batch: dict
            Input dictionary of batch data. It provides `n_qcluster`, `n_flash`
            and `pairs` for the loss matrix.

        Returns
        -------
        match_output: dict
            Output dictionary for the matching results.

            `tspent`: float
                Time spent in seconds.
            `idx`: list of (int,int)
                Indcies for the matched (charge, flash) pair.
                `None` if the bipartite match fails.
            `loss_matrix`: tensor
                Loss matrix of all pairs. Filtered pair (not included in the
                fit) has a value of `inf`.
            `dx_best`: tensor
                Best dx estimation for the matched pair.
                `nan` for the clusters without matching flash.
                `None` if the bipartite match fails.
            `pe_best`: tensor
                Best reco. pe for the match matched pair.
                `nan` for the clusters without matching flash.
                `None` if the bipartite match fails.
        '''
        t_start = time.time()

        # fill loss matrix
        n_q = batch['n_qcluster']
        n_f = batch['n_flash']
        pairs = torch.stack(batch['pairs']).cpu()

        # clone loss_matrix from prefit
        # create if not exisit (default value = inf)
        loss_matrix = output.get('prefit', {}).get('loss_matrix')
        if loss_matrix is None:
            loss_matrix = torch.full((n_q, n_f), torch.inf)
        else:
            loss_matrix = loss_matrix.clone()

        # update loss_matrix
        loss_matrix[pairs[0],pairs[1]] = output['fit']['loss_best']

        # bipartite match
        try:
            idx = linear_sum_assignment(loss_matrix.nan_to_num())
        except ValueError:
            self.print('Fail to do bipartite match!')
            idx = None

        # fill matched dx and pe 
        if idx is not None:
            # mapping from (i_q, i_f) -> 1-dim pair index 
            m_pairs = torch.sparse_coo_tensor(
                pairs, torch.arange(pairs.size(1))
            )

            # selected (matched) pairs 
            sel = torch.tensor([m_pairs[i_q,i_f] for i_q,i_f in zip(*idx)])

            dx_best = torch.full((n_q,), torch.nan)
            dx_best[idx[0]] = output['fit']['dx_best'][sel]

            loss_best = torch.full((n_q,), torch.nan)
            loss_best[idx[0]] = output['fit']['loss_best'][sel]

            pe_sel = output['fit']['pe_best'][sel]
            pe_best = torch.full((n_q,pe_sel.size(1)), torch.nan)
            pe_best[idx[0]] = pe_sel
        else:
            dx_best = None
            pe_best = None
            loss_best = None

        t_stop = time.time()

        match_output = {
            'tspent': t_stop - t_start,
            'idx': idx,
            'loss_matrix': loss_matrix,
            'dx_best': dx_best,
            'pe_best': pe_best,
            'loss_best': loss_best,
        }
        return match_output

    def _fit(self, batch, dx_init=None):
        '''
        An auxiliary ftting function  for batch data.

        Arguments
        ---------
        batch: dict
            See ``make_input_batch()``.

        dx_init: array_like
            Initial values for x-offset. Usually obtained from the output of
            ``prefit()`` or ``prefilter()``.

        Returns
        -------
        output: dict
            Fit results as a dictionary.

            `loss_hist`, `dx_hist` : tensor
                History of loss (dx) for all iterations.

            `loss_best`, `dx_best`, `pe_best` : tensor
                Best loss (dx, reco. p.e.) among iterations.

            `stuck_cnts`: tensor
                Number of iterations of no improvement in loss and dx.

            `niters`: int
                Number of iterations for the fit.

            `tspent`: float
                Time spent in seconds on the fitting procedure.
        '''
        t_start = time.time()

        # hyperparameters, early stop conditioins etc.
        device = self.device
        max_iters = self.max_iterations
        stop_loss_delta = self.stop_loss_delta
        stop_dx_delta = self.stop_xmin_delta
        stop_patience = self.stop_patience

        # init optimizer
        model = MultiFlashHypothesis(
            self._vis_model, batch['dx_ranges'], dx_init,
        ).to(device)
        self.print(f'[fit] Optmization for {len(model)} hypotheses')

        # Note: avoid putting `model.parameters()` in the optimizer,
        # as the `SirenVis` paramters may also be updated if not frozen.
        # We only want dx as optimizible parameters.
        optimizer = torch.optim.Adam(model.pars, lr=self.initial_lr)
        scheduler = scheduler_factory(
            optimizer, self.cfg.get('Scheduler'), self.verbose
        )

        # masks (for early stopping)
        n_clust = len(model)
        mask_clust = torch.full((n_clust,), True, device=device)

        # data input
        qcluster_batch = batch['qclusters']
        qcluster_size = batch['sizes']
        flash_batch = batch['flashes']

        dx_hist = torch.zeros((max_iters, n_clust), device=device)
        loss_hist = torch.zeros_like(dx_hist)
        stuck_cnts = torch.zeros(n_clust, dtype=torch.int, device=device)
        loss_best = torch.full((n_clust,), torch.inf, device=device)
        dx_best = torch.empty_like(loss_best)
    
        for i_iter in trange(max_iters, disable=not self.verbose):
            if ~mask_clust.any():
                self.print('Early termination at iteration', i_iter)
                break
            
            out = model(qcluster_batch, qcluster_size, mask_clust)
            tgt = flash_batch[mask_clust]
            
            loss_pairs = self.crit(out, tgt)
            loss = loss_pairs.sum()
            
            # backward propagation
            # for updating requires_grad inside optimization
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # update history
            with torch.no_grad():
                dx_hist[i_iter] = model.dx.cpu()
                loss_hist[i_iter,mask_clust] = loss_pairs
            if i_iter > 0:
                loss_hist[i_iter,~mask_clust] = loss_hist[i_iter-1,~mask_clust]
            
            # update best fit
            loss_curr = loss_hist[i_iter]
            dx_curr = dx_hist[i_iter]
            
            mask_best = loss_curr < loss_best
            mask_reset = loss_best - loss_curr > stop_loss_delta
            dx_best[mask_best] = dx_curr[mask_best]
            loss_best[mask_best] = loss_curr[mask_best]
                                
            # update stuck counters
            if i_iter > 0:
                dx_lass = dx_hist[i_iter-1]
                mask_reset |= abs(dx_curr-dx_best) > stop_dx_delta
                stuck_cnts[mask_reset] = 0
                stuck_cnts[~mask_reset] += 1
                
            # update for early-stopping matched pair
            mask_clust = stuck_cnts < stop_patience
            for par in compress(model.pars, ~mask_clust):
                par.requires_grad = False

            # update scheduler
            if scheduler is not None:
                if 'metrics' in scheduler.step.__code__.co_varnames:
                    scheduler.step(loss)
                else:
                    scheduler.step()
                
        # save best reco pe
        model.dx = dx_best
        with torch.no_grad():
            pe_best = model(qcluster_batch, qcluster_size)

        t_stop = time.time()

        # save fit results
        output = {
            'dx_hist': dx_hist[:i_iter].cpu(),
            'loss_hist': loss_hist[:i_iter].cpu(),
            'stuck_cnts': stuck_cnts[:i_iter].cpu(),
            'tspent': t_stop - t_start,
            'niter': i_iter,
            'dx_best': dx_best.cpu(),
            'loss_best': loss_best.cpu(),
            'pe_best': pe_best.cpu(),
        }

        return output

    def fit(self, input : FlashMatchInput):
        '''
        The Main fitting function.

        Arguments
        ---------
        input: FlashMatchInput

        Returns
        -------
        output: dict
            Fit results in a dictionay. Each stage is stored under its own
            sub-dictionary.

            `prefit` : dict, optional
                Prefit results. See output of ``prefit()`` or ``prefilter()``
                for non-matching (matching) mode. Absent if `Prefit = False` in
                the config.

            `fit`: dict
                Fit resuts. See output of ``_fit()``.

                For matching mode, the outputs are the best fits for all
                potential charge-flash pairs. See `match` output for the
                matched pairs.

                For non-matching mode, the best fits have same indices as input
                qclusters.

            `match`: dict, optional
                Matching results. Perform bipartite matching for all potential
                pairs. See ``match()`` output. Absent in non-matching mode; use
                `fit`.  output instead. 

            `tpsent`: float
                Total time spent in seconds for all fitting procedures (prefit
                + fit + match).
        '''

        # gather qcluster and flash tensors from FlashMatchInput instance
        data = self.collate_input(input)
        qclusters, flashes = \
            tuple(map(data.__getitem__, ['qclusters','flashes']))

        output = {'tspent': 0.}
        pairs = None
        dx_init = None
        if self.do_prefit:
            prefit_func = self.prefilter if self.do_match else self.prefit
            prefit = prefit_func(qclusters, flashes)

            pairs = prefit.get('pairs', None)
            dx_init = prefit['dx_init']
            output['prefit'] = prefit
            output['tspent'] += prefit['tspent']
        else:
            self.print('[fit] set "Prefit: True" for better convergence')

        if self.do_match and not self.do_prefit:
            # take all pair combinations
            self.print('[fit] set "Prefilter: True" for faster running time')
            pairs = self.pair_product(len(qclusters), len(flashes), self.device)

        # make batch input and execute _fit()
        batch = self.make_input_batch(data, pairs=pairs)

        fit = self._fit(batch, dx_init=dx_init)
        output['fit'] = fit
        output['tspent'] += fit['tspent']

        # pair matching
        if self.do_match:
            match = self.match(output, batch)
            output['match'] = match
            output['tspent'] += match['tspent']

        return output


    def simple_fit(self, input : FlashMatchInput):
        '''
        Flash mathcing without x-optimization.

        Arguments
        ---------
        input: FlashMatchInput

        Returns
        -------
        output: dict

            `tspent`: float
                Time spent in seconds for the matching

            `idx`: list of (int, int)
                Indices for the matched (charge, flash) pairs.
                `None` if the bipartite match fails.

            `loss_matrix`: tensor
                Loss matrix of all pair combinations.

            `pe_pred`: tensor
                Predicted p.e. from input charge clusters.
        '''
        
        t_start = time.time()
        
        batch = self.make_input_batch(input)
        with torch.no_grad():
            pred = MultiFlashHypothesis.apply(
                0, batch['qclusters'], batch['sizes'], self._vis_model
            )

        obs = batch['flashes']

        loss_matrix = self.crit(pred.unsqueeze(0), obs.unsqueeze(1)).cpu()

        try:
            idx = linear_sum_assignment(loss_matrix)
        except ValueError:
            self.print('Fail to do biparitite match!')
            idx = None

        t_stop = time.time()

        output = {
            'tspent': t_stop - t_start,
            'idx': idx,
            'loss_matrix': loss_matrix,
            'pe_pred': pred.cpu(),
        }

        return output
