"""Run the physical constant-body counterpart with historical matched settings."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import sys, json, hashlib, copy, argparse
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).absolute().parent
ROOT=Path('G:/2026.3/代码/paper_results/geology_v4_20260922')
CODE=Path('F:/2026.3/代码')

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def matched_config(old):
    c=copy.deepcopy(old)
    c.update(model_id='geology_v4_constant_body_10000',density_law='matched_per_body_mean')
    return c

def validate_ids(config,rows):
    for key,split in [('train_ids','train'),('validation_ids','validation')]:
        ids=config[key]
        if not ids or len(ids)!=len(set(ids)):raise ValueError('Empty/duplicate '+key)
        if any(not isinstance(i,int) or i<0 or i>=len(rows) or rows[i]['split']!=split for i in ids):
            raise ValueError('Invalid split membership '+key)
    if set(config['train_ids'])&set(config['validation_ids']):raise ValueError('Overlap')

def paths_for(out,epochs=120):
    sys.path.insert(0,str(CODE))
    from paired_density_c0c1_contract import PilotPaths
    return PilotPaths(run_dir=out,primary_weights=out/f'epoch_{epochs:03d}_weights.pth',
        best_weights=out/'best_validation_weights.pth',last_checkpoint=out/'last_checkpoint.pth',
        history=out/'history.npz',metadata=out/'metadata.json',failure=out/'failure.json',
        snapshot_weights={epochs:out/f'epoch_{epochs:03d}_weights.pth'})

def run(resume=False,stop_after_epoch=None):
    data=HERE/'constant_reference_11600';out=HERE/'constant_train10000_seed20260727'
    if out.exists() and not resume:raise FileExistsError('Refuse overwrite; use --resume')
    if resume and not (out/'last_checkpoint.pth').is_file():raise FileNotFoundError('Resume checkpoint absent')
    prep=json.loads((data/'preparation.json').read_text(encoding='utf8'))
    for n,h in prep['output_hashes'].items():assert sha(data/n)==h,n
    for n,h in prep['source_hashes'].items():assert sha(ROOT/'dataset_11600'/n)==h,n
    oldpath=ROOT/'train10000_seed20260727/metadata.json'
    old=json.loads(oldpath.read_text(encoding='utf8'));contract=old['resume_contract']
    assert old['status']=='complete' and old['completed_epochs']==120
    ready=json.loads((ROOT/'readiness.json').read_text(encoding='utf8'))
    source_hashes={}
    for name in ['train_paired_density_c0c1.py','train_formal_20k.py','paired_density_c0c1_contract.py']:
        h=sha(CODE/name)
        assert h==ready['snapshot_files']['historical_training_sources\\'+name],name
        source_hashes[name]=h
    assert source_hashes['train_paired_density_c0c1.py']==contract['training_source_sha256']
    init=CODE/'experiments/c0c1_paired_v2_r2/initialization/seed_20260727.pth'
    assert sha(init)==contract['initialization_sha256']
    hweights=ROOT/'train10000_seed20260727/epoch_120_weights.pth'
    assert sha(hweights)==old['artifact_sha256']['primary_weights']
    config=matched_config(old['config'])
    rows=json.loads((data/'source_metadata.json').read_text(encoding='utf8'))
    validate_ids(config,rows)
    assert len(config['train_ids'])==10000 and len(config['validation_ids'])==1200
    assert config['max_epochs']==120 and not config['early_stopping']
    assert torch.cuda.is_available(),'CUDA required'
    images=np.load(data/'gravity.npy');targets=np.load(data/'density.npy')
    assert images.shape==(11600,1,32,32) and targets.shape==(11600,16,32,32)
    assert np.isfinite(images).all() and np.isfinite(targets).all()
    identity=dict(adapter_sha256=sha(__file__),preparation_sha256=sha(data/'preparation.json'),
        original_metadata_sha256=sha(oldpath),initialization_sha256=sha(init),
        training_sources=source_hashes,heterogeneous_weights_sha256=sha(hweights),
        evaluation_protocol_sha256=sha(HERE/'配对训练评估协议_20261016.md'))
    sys.path.insert(0,str(CODE))
    from train_paired_density_c0c1 import execute_training
    state=torch.load(init,map_location='cpu',weights_only=True)
    torch.set_num_threads(2)
    print('Verified matched 10000/1200 split, source and initialization; CUDA training starts.',flush=True)
    execute_training(config,paths_for(out),images,targets,state,identity,device='cuda',
                     num_workers=0,resume=resume,stop_after_epoch=stop_after_epoch)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');p.add_argument('--stop-after-epoch',type=int)
    args=p.parse_args();run(args.resume,args.stop_after_epoch)
