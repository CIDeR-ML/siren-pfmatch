from pfmatch.__future__ import datasets
from pfmatch.__future__ import loaders

def dataset_factory(cfg):
    ds_cfg  = cfg['data']['dataset'].copy()
    ds_type = ds_cfg.pop('interface')
    if not getattr(datasets,ds_type):
        raise TypeError(f'data interface type {ds_type} not available')
    
    filepath = ds_cfg.pop('filepath')
    if isinstance(filepath, str):
        filepath = [filepath]

    from glob import glob
    files = []
    for fpath in filepath:
        files.extend(glob(fpath))

    ds_cfg['filepath'] = files
    return getattr(datasets,ds_type)(**ds_cfg)

def loader_factory(cfg):

    ds = dataset_factory(cfg)

    data_cfg = cfg['data']
    loader_cfg = data_cfg.get('loader', dict(interface='DataLoader')).copy()
    loader_type = loader_cfg.pop('interface')
    if loader_type == 'DataLoader':
        loader_cfg['collate_fn']=ds.read_many
    loader = getattr(loaders,loader_type)(ds, **loader_cfg)
    if loader_type == 'DataLoader':
        loader_cfg['collate_fn']='ds.read_many'
    return loader

