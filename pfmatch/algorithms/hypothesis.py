from __future__ import annotations
from itertools import compress

import torch
import numbers
from photonlib import PhotonLib, MultiLib
from slar.nets import SirenVis, MultiVis

def shift_x_in_chunk(coords, sizes, dx):
    """
    Shift x -> x + dx, allowing dx = 0 or None.
    
    Arguments
    ---------
    corrds : Tensor (N,3) 
        Input coordinates
    sizes: Tensor (M,)
        Split size of coords into M chunks.
    dx: Tensor (M,) | 0 | None
        Offset in x, one per chunk. Do nothing if dx = 0 or None.


    Returns
    -------
    coords_new : Tensor (N,3)
        Output cooridinates
    """

    if dx is None:
        return coords

    if isinstance(dx, numbers.Real) and dx==0:
        return coords

    x = coords[:,0] + dx.repeat_interleave(sizes)
    coords_new = torch.column_stack([x, coords[:,1:]])
    return coords_new

class F(torch.autograd.Function):
    """
    Custom autograd function for PhotonLib
    """

    @staticmethod
    def forward(ctx, track, plib):
        ctx.save_for_backward(track)
        ctx.plib = plib
        pe_v = torch.sum(plib.visibility(track[:,:3])*(track[:, 3].unsqueeze(-1)), axis = 0)   
        return pe_v

    @staticmethod
    def backward(ctx, grad_output):
        track = ctx.saved_tensors[0]
        grad_plib = ctx.plib.gradx(track[:,:3])
        grad_input = torch.matmul(grad_plib*track[:,3].unsqueeze(-1), grad_output.unsqueeze(-1))
        pad = torch.zeros(grad_input.shape[0], 3, device=grad_output.device)
        return torch.cat((grad_input,pad), -1), None


class FlashHypothesis(torch.nn.Module):

    def __init__(self, cfg:dict, plib:PhotonLib | MultiVis | SirenVis | MultiVis): 
        super().__init__()
        self._plib = plib
        # fmatch_cfg = cfg.get('FlashHypothesis',dict())

        self._dx = torch.nn.Parameter(torch.as_tensor([0.]))
        self._dx.data.fill_(0.)

        self._xmin = self.plib.meta.ranges[0,0].item()
        self._xmax = self.plib.meta.ranges[0,1].item()

        self._dx_min = self._xmin
        self._dx_max = self._xmax

    @property
    def plib(self):
        return self._plib

    @property
    def dx(self):
        return self._dx.item()

    @dx.setter
    def dx(self, value):
        return self._dx.data.fill_(value)

    def dx_range(self, track):
        track_xmin = torch.min(track[:,0]).item()
        track_xmax = torch.max(track[:,0]).item()
        assert self._xmin <= track_xmin, f'Track minimum x {track_xmin} is outside the volume minimum x {self._xmin}'
        assert self._xmax >= track_xmax, f'Track maximum x {track_xmax} is outside the volume maximum x {self._xmax}'

        dx_min = self._xmin - track_xmin
        dx_max = self._xmax - track_xmax
        return dx_min, dx_max
        
    def forward(self,track: torch.Tensor):
        """
        fill flash hypothsis based on given qcluster track
        ---------
        Arguments
        track: qcluster track of 3D position + charge
        ---------
        Returns hypothesized number of p.e. to be detected in each PMT
        """

        self._dx_min, self._dx_max = self.dx_range(track)
        self._dx.data.clamp_(self._dx_min, self._dx_max)
        shift = torch.cat((self._dx, torch.zeros(3, device=track.device)), -1)
        shifted_track = torch.add(track, shift.expand(track.shape[0], -1))
        if isinstance(self.plib,PhotonLib) or isinstance(self.plib,MultiLib):
            return F.apply(shifted_track,self.plib)
        elif isinstance(self.plib,SirenVis) or isinstance(self.plib,MultiVis):
            #x = self.plib.meta.norm_coord(shifted_track[:,:3])
            #pe_v = torch.sum(self.plib._inv_xform_vis(self.plib(x))*(shifted_track[:, 3].unsqueeze(-1)), axis = 0)
            pe_v = torch.sum(self.plib.visibility(shifted_track[:,:3])*(shifted_track[:,3].unsqueeze(-1)),axis=0)
            return pe_v
        else:
            raise TypeError('Unsupported type(vis_mod)', type(self.plib))


class PLibPrediction(torch.autograd.Function):
    '''
    Autograd function for PhotonLib. Apply for batch input of multiple charge
    clusters.
    '''
    
    @staticmethod
    def forward(ctx, dx, batch, sizes, plib):
        coords = shift_x_in_chunk(batch[:,:3], sizes, dx)
    
        q = batch[:,3]
        
        ctx.save_for_backward(coords,q)
        ctx.plib = plib
        ctx.sizes = sizes
        
        pred = torch.segment_reduce(
            plib.visibility(coords) * q.unsqueeze(-1),
            'sum',
            lengths=sizes,
            axis=0
        )

        return pred
            
    @staticmethod
    def backward(ctx, grad_output):
        coords, q = ctx.saved_tensors
        plib = ctx.plib
        sizes = ctx.sizes
        
        grad_pred_x = torch.segment_reduce(
            plib.gradx(coords) * q.unsqueeze(-1),
            'sum',
            lengths=sizes,
            axis=0
        )

        grad_x = (grad_pred_x * grad_output).sum(axis=1)

        return grad_x, None, None, None

class MultiFlashHypothesis(torch.nn.Module):
    '''
    Flash hypothesis for batch input that contains multiple charge clusters.
    Each charge cluster has its own optimizible x-offset.

    Arguments
    ---------
    vis_model : PhotonLib | SirenVis
        Visibilty model in form of PhotonLib (LUT) or SirenVis.

    dx_ranges: array_like (N, 2)
        List of x-offset ranges for _N_ flash hypothesis.

    dx_init: array_like (N,), optional
        Initial x-offset for _N_ flashes. Default: 0.
        `len(dx_init) == len(dx_ranges)`
    '''
    def __init__(self, 
                 vis_model : PhotonLib | MultiLib | SirenVis | MultiVis, 
                 dx_ranges, 
                 dx_init=None,
                 ):
        super().__init__()
        self.vis_model = vis_model
        self._dx_ranges = torch.as_tensor(dx_ranges)

        if dx_init is None:
            dx_init = torch.zeros(len(self))

        dx_init = torch.as_tensor(dx_init)
        if len(dx_init) != len(self):
            raise ValueError(f'len(dx_init) != len(dx_ranges)')

        self.pars = torch.nn.ParameterList([
            torch.nn.Parameter(dx0) for dx0 in dx_init
        ])

    def __len__(self):
        return len(self._dx_ranges)
    
    def mask_dx(self, mask=None, clamp=False):
        '''
        Offset parameters with optional mask.

        Arguments
        ---------
        mask: Tensor(bool), optional
            Mask to be applied. Default: None.
            If `None`, returns all parameters.

        clamp: bool, optional
            Clamp offset parameters within ``_dx_ranges``.
            Default: `False`

        Returns
        -------
        dx: Tensor, (_N_,)
            Offset parameters. If masked, `N <= len(self)`.
            Otherwise, `N == len(self)`.
        '''
        dx_ranges = self._dx_ranges
        if mask is None:
            pars = torch.stack(list(self.pars))
        else:
            dx_ranges = dx_ranges[mask]
            pars = torch.stack(list(compress(self.pars, mask)))
            
        if clamp:
            pars = pars.clamp(*dx_ranges.T)
            
        return pars
               
    @property
    def dx(self):
        '''
        Offset parameters without mask and clamp.
        '''
        return self.mask_dx()
    
    @dx.setter
    def dx(self, values):
        if len(self.pars) != len(values):
            raise ValueError(f'len(values) != len(pars)')

        for p, v in zip(self.pars, values):
            p.data.fill_(v)

    @staticmethod
    def apply(dx, batch, sizes, vis_model):
        '''
        An auxiliary function to apply x-offset to multiple charge clusters,
        either using `PhotonLib` or `SirenVis`.

        Arguments
        ---------

        dx: tensor
            Offset in x to be applied to the charge clusters.
            One offset per clusters.

        batch: tensor
            A sigle tensor of the charge points `(x,y,z,q)` for all clusters.

        sizes: tensor
            Number of charge points per cluster.
            `len(sizes)` gives the number of charge clusters, and `sum(sizes)
            == len(batch)` represents the total number of charge points. 

        vis_model: PhotonLib | SirenVis
            Visibility model. The gradient w.r.t. dx is provided by
            `PLibPrediction` in case of `PhotonLib`.

        Returns
        -------
        output: Tensor, (_N_,_M_)
            Optical output prediction with the current x-offsets.
            If masked, `N` is the number of masked clusters. Otherwise `N ==
            len(self)` for all clusters.
            `M` is the number of the PMTs.

        '''
        if isinstance(vis_model, PhotonLib) or isinstance(vis_model, MultiLib):
            output = PLibPrediction.apply(dx, batch, sizes, vis_model)

        elif isinstance(vis_model, SirenVis) or isinstance(vis_model, MultiVis):
            coords = shift_x_in_chunk(batch[:,:3], sizes, dx)

            q = batch[:,3]

            output = torch.segment_reduce(
                vis_model.visibility(coords) * q.unsqueeze(-1),
                'sum',
                lengths=sizes,
                axis=0,
            )

        else:
            raise TypeError('Unsupported type(vis_mod)', type(vis_model))
        return output
        
    def forward(self, batch, sizes, mask=None):
        '''
        Predict optical output for batch input of charge clusters.

        Arguments
        ---------
        batch: Tensor
        sizes: Tensor
            See input of ``self.apply()``.

        mask: Tensor(bool), optional
            Mask to be applied. Default `None`.
            If mask is not set, all clusters in `batch` are evalulated.
            Otherwise, `len(mask) == len(sizes)`, only the masked clusters are
            included.

        Returns
        -------
        output: Tensor, (_N_,_M_)
            See output of ``self.apply()``.
        '''

        dx = self.mask_dx(mask, clamp=True)
        
        if mask is not None:
            batch = batch[mask.repeat_interleave(sizes)]
            sizes = sizes[mask]
            
        output = self.apply(dx, batch, sizes, self.vis_model)

        return output
