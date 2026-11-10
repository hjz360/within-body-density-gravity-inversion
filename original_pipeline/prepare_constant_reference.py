"""Prepare self-consistent constant-per-body counterpart; never train/overwrite."""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, hashlib, time, shutil
from pathlib import Path
import numpy as np
import h5py
from threadpoolctl import threadpool_limits

ROOT=Path('G:/2026.3/代码/paper_results/geology_v4_20260922')
OUT=Path(__file__).absolute().parent/'constant_reference_11600'

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def constant(t,l):
    if t.shape!=l.shape or not np.isfinite(t).all() or not np.array_equal(t>0,l>0):
        raise ValueError('Invalid density/support')
    c=np.zeros_like(t,dtype=np.float32)
    for k in np.unique(l):
        if k:
            mask=l==k;c[mask]=np.mean(t[mask],dtype=np.float64)
    return c

def tests():
    t=np.array([0.,.4,.8,1.,.6],np.float32);l=np.array([0,1,1,2,2])
    c=constant(t,l)
    np.testing.assert_allclose(c,[0,.6,.6,.8,.8],atol=1e-7)
    np.testing.assert_array_equal(constant(c,l),c)
    for bad in [np.array([0,1,0,2,2]),np.array([1,1,1,2,2])]:
        try:constant(t,bad)
        except ValueError:pass
        else:raise AssertionError('Invalid support accepted')
    print('Manufactured invariants passed',flush=True)

def main():
    tests();start=time.time();src=ROOT/'dataset_11600'
    if OUT.exists():raise FileExistsError('Refuse overwrite: '+str(OUT))
    assert shutil.disk_usage(OUT.parent).free>3*1024**3
    audit=json.loads((src/'audit.json').read_text(encoding='utf8'))
    for name,digest in audit['files'].items():
        assert sha(src/name)==digest,name
    kernel=Path('F:/2026.3/代码/G.mat')
    assert sha(kernel)==audit['identity']['kernel_sha256']
    rows=json.loads((src/'metadata.json').read_text(encoding='utf8'))
    assert len(rows)==11600 and len({r['group_key'] for r in rows})==11600
    t=np.load(src/'density.npy');l=np.load(src/'body_labels.npy');d=np.load(src/'gravity.npy')
    assert t.shape==l.shape==(11600,16,32,32) and d.shape==(11600,1,32,32)
    with h5py.File(kernel) as f:g=np.asarray(f['G'],dtype=np.float64).T*1000
    assert g.shape==(1024,16384)
    c=np.zeros_like(t);maxmass=0.;count=0
    for i in range(len(t)):
        assert hashlib.sha256((l[i]>0).tobytes()).hexdigest()==rows[i]['support_sha256']
        c[i]=constant(t[i],l[i])
        for k in np.unique(l[i]):
            if k:
                mask=l[i]==k;a=t[i][mask].sum(dtype=np.float64);b=c[i][mask].sum(dtype=np.float64)
                maxmass=max(maxmass,abs(a-b)/a);count+=1
        np.testing.assert_array_equal(c[i]>0,t[i]>0)
    assert maxmass<1e-6
    dc=np.empty_like(d);maxold=0.;maxnew=0.
    with threadpool_limits(2):
        for begin in range(0,len(t),64):
            end=min(begin+64,len(t))
            original=g@t[begin:end].reshape(end-begin,-1).astype(float).T
            exact=g@c[begin:end].reshape(end-begin,-1).astype(float).T
            maxold=max(maxold,float(np.max(abs(original.T-d[begin:end].reshape(end-begin,-1)))))
            dc[begin:end]=exact.T.reshape(-1,1,32,32)
            maxnew=max(maxnew,float(np.max(abs(exact.T-dc[begin:end].reshape(end-begin,-1)))))
            if begin%1024==0:print('Forward checked',end,'/11600',flush=True)
    assert maxold<2e-5 and maxnew<2e-5
    OUT.mkdir()
    np.save(OUT/'density.npy',c);np.save(OUT/'gravity.npy',dc)
    shutil.copyfile(src/'metadata.json',OUT/'source_metadata.json')
    report=dict(status='prepared_not_training_ready',source=str(src),count=len(t),body_count=count,
        source_hashes=audit['files'],kernel_sha256=sha(kernel),script_sha256=sha(__file__),
        max_relative_body_mass_error=maxmass,max_source_forward_error_mgal=maxold,
        max_float32_forward_rounding_mgal=maxnew,
        output_hashes={n:sha(OUT/n) for n in ['density.npy','gravity.npy','source_metadata.json']},
        elapsed_seconds=time.time()-start,
        semantics='C uses per-body mean density and G(C); H remains original. source_metadata density statistics describe H only.',
        remaining=['Independent re-read and forward check','Training configuration and checkpoint provenance','Freeze common test and selection rules'],
        test_scope='Existing 400 cases are previously viewed; not a new blind test')
    (OUT/'preparation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
