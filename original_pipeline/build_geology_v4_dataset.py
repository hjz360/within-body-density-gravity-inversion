"""G-drive dataset generation with split leakage checks and resumable chunks."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import sys,json,time,hashlib,shutil
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from threadpoolctl import threadpool_limits
from geology_v4 import FAMILIES,NAMES,EXPECTED_COMPONENTS,make_plan,model

SOURCES=['geology_v4.py','geology_v3.py','geology_geometry_v2.py','geology_gallery.py','build_geology_v4_dataset.py']

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def write(p,value):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)

def audit_splits(rows):
    splits=sorted(set(r['split'] for r in rows));pairs=[]
    if len({r['group_key'] for r in rows})!=len(rows):raise ValueError('Repeated geometry group')
    for i,s in enumerate(splits):
        a={r['support_sha256'] for r in rows if r['split']==s}
        for t in splits[i+1:]:
            b={r['support_sha256'] for r in rows if r['split']==t}
            overlap=len(a&b);pairs.append(dict(left=s,right=t,identical_supports=overlap))
            if overlap:raise ValueError('Identical support crosses split')
    return dict(passed=True,pairs=pairs)

def worker(row):
    m,l,a=model(row)
    assert m.shape==(16,32,32) and np.isfinite(m).all()
    assert float(m.min())>=0 and float(m.max())<=1.1
    assert np.array_equal(m>0,l>0) and a['passed']
    assert not any(a['boundary_flags'].values()) and all(x==1 for x in a['body_volume_retention'])
    assert len(a['body_voxels'])==EXPECTED_COMPONENTS[row['family']]
    points=np.argwhere(l>0)
    record=dict(row,geometry_audit=a,support_sha256=hashlib.sha256((l>0).tobytes()).hexdigest(),
        support_centroid_zyx_m=((points.mean(0)+.5)*50).tolist(),
        support_extent_zyx_m=((points.max(0)-points.min(0)+1)*50).tolist(),
        density_mean_inside=float(m[l>0].mean()),density_max=float(m.max()),
        mass_kg=float(m.astype(float).sum()*1000*50**3))
    return m,l,record

def run(out,smoke=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    plan=make_plan(100,0,0) if smoke else make_plan()
    identity=dict(plan_sha256=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest(),sources={f:sha(Path(__file__).parent/f) for f in SOURCES},kernel_sha256=sha(r'F:\2026.3\代码\G.mat'))
    n=len(plan);start=0;mode='w+'
    if (out/'progress.json').exists():
        previous=json.loads((out/'progress.json').read_text(encoding='utf-8'))
        assert previous['identity']==identity,'Input identity changed: refuse resume'
        if previous['status']=='complete':print('Dataset already complete',flush=True);return
        start=previous['completed'];mode='r+'
    else:
        assert not any(out.iterdir()),'Do not overwrite an existing dataset'
    assert shutil.disk_usage(out).free>3*1024**3,'Insufficient G drive free space'
    density=np.lib.format.open_memmap(out/'density.npy',mode=mode,dtype=np.float32,shape=(n,16,32,32))
    labels=np.lib.format.open_memmap(out/'body_labels.npy',mode=mode,dtype=np.uint8,shape=(n,16,32,32))
    gravity=np.lib.format.open_memmap(out/'gravity.npy',mode=mode,dtype=np.float32,shape=(n,1,32,32))
    records=json.loads((out/'metadata.json').read_text(encoding='utf-8'))[:start] if start else []
    write(out/'group_plan.json',plan)
    src=out/'source_snapshot';src.mkdir(exist_ok=True)
    for name in SOURCES:shutil.copy2(Path(__file__).parent/name,src/name)
    sys.path.insert(0,r'F:\2026.3\代码')
    from gravity_validation import load_hdf5_kernel,rectangular_prism_gz,observation_grid
    G=load_hdf5_kernel(r'F:\2026.3\代码\G.mat')*1000
    xo,yo,zo=observation_grid();spot=[]
    for z,y,x in [(1,3,27),(8,23,7),(14,15,17)]:
        independent=rectangular_prism_gz(xo,yo,zo,(x*50,(x+1)*50,y*50,(y+1)*50,z*50,(z+1)*50),1000)
        error=float(np.max(abs(independent-G[:,z*1024+y*32+x])));spot.append(error)
        assert np.allclose(independent,G[:,z*1024+y*32+x],rtol=1e-5,atol=1e-7)
    begin=time.perf_counter()
    with threadpool_limits(limits=2),ProcessPoolExecutor(max_workers=4) as pool:
        for low in range(start,n,100):
            high=min(low+100,n)
            result=list(pool.map(worker,plan[low:high],chunksize=5))
            batch=np.array([x[0] for x in result]);density[low:high]=batch
            labels[low:high]=np.array([x[1] for x in result]);records.extend(x[2] for x in result)
            gravity[low:high]=(G@batch.reshape(high-low,-1).T).T.reshape(-1,1,32,32)
            density.flush();labels.flush();gravity.flush();write(out/'metadata.json',records)
            write(out/'progress.json',dict(status='generating',completed=high,total=n,identity=identity,elapsed_current_run_seconds=time.perf_counter()-begin))
            print(f'Generated {high}/{n}, elapsed {time.perf_counter()-begin:.1f}s',flush=True)
        split_audit=audit_splits(records)
        err=0
        for low in range(0,n,100):
            high=min(low+100,n)
            g=(G@np.asarray(density[low:high],dtype=float).reshape(high-low,-1).T).T.reshape(-1,1,32,32)
            err=max(err,float(np.max(abs(g-gravity[low:high]))))
        assert err<1e-5 and np.isfinite(gravity).all() and gravity.min()>0
    stats={}
    for split in sorted(set(r['split'] for r in records)):
        rr=[r for r in records if r['split']==split];centers=np.array([r['support_centroid_zyx_m'] for r in rr])
        stats[split]=dict(n=len(rr),families=dict(Counter(r['family'] for r in rr)),laws=dict(Counter(r['density_law_index'] for r in rr)),
            components=dict(Counter(EXPECTED_COMPONENTS[r['family']] for r in rr)),centroid_zyx_quantiles_m=np.quantile(centers,[0,.05,.5,.95,1],axis=0).tolist(),
            boundary_contact_count=sum(r['geometry_audit']['boundary_contact'] for r in rr))
    audit=dict(passed=True,identity=identity,counts=stats,split_audit=split_audit,
        forward_roundtrip_max_abs_mgal=err,independent_prism_spotcheck_max_abs_mgal=spot,
        complete_body_retention=True,attempts=sum(r['geometry_audit']['attempts'] for r in records),
        elapsed_seconds=time.perf_counter()-begin,
        files={f:sha(out/f) for f in ['density.npy','gravity.npy','body_labels.npy','metadata.json','group_plan.json']},
        limitations=['50m voxels; no subcell integration','Old six families retain old placement scheme; four new ones use feasible translations','Geometry and sample count changed together','Sealed test is in-distribution, not unseen topology','Positive density contrasts only; not field-calibrated'])
    write(out/'audit.json',audit);write(out/'progress.json',dict(status='complete',completed=n,total=n,identity=identity))
    print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':run(sys.argv[1],smoke='--smoke' in sys.argv[2:])
