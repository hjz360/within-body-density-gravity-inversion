"""One-shot frozen-weight same-generator follow-up, protocol fixed beforehand."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import sys,json,csv
from pathlib import Path
import numpy as np
import h5py,torch
from threadpoolctl import threadpool_limits
from train_constant_reference import HERE,ROOT,CODE,sha
from prepare_constant_reference import constant
sys.path.insert(0,str(HERE.parent))
from geology_v4 import FAMILIES,model
from internal_recovery_audit import recovery

def main():
    out=HERE/'frozen_followup_200_seed20260727'
    if out.exists():raise FileExistsError(out)
    protocol=HERE/'新增冻结网络评价协议_20261016.md'
    audit=json.loads((ROOT/'dataset_11600/audit.json').read_text(encoding='utf8'))
    for name,h in audit['identity']['sources'].items():assert sha(HERE.parent/name)==h,name
    old=json.loads((ROOT/'dataset_11600/metadata.json').read_text(encoding='utf8'))
    oldkeys={r['group_key'] for r in old};oldsha={r['support_sha256'] for r in old}
    weights=[HERE/'constant_train10000_seed20260727/epoch_120_weights.pth',ROOT/'train10000_seed20260727/epoch_120_weights.pth']
    assert sha(weights[0])=='f816526e22d835f9779eadb6e17684dfbba469e3433a40534e9904bcaa69688f'
    assert sha(weights[1])=='1831249cd9aa01b698dd2260c0fca605d2d0cd7e37eca677a06b82c4d190212f'
    import hashlib
    c=[];h=[];labs=[];meta=[];newsha=set();newkeys=set()
    for i in range(200):
        row=dict(family=FAMILIES[i%10],geometry_index=9000000+i//10,
                 density_seed_index=12000000+i,density_law_index=(i//10)%5)
        t,l,a=model(row);key=f"{row['family']}:{row['geometry_index']}"
        supporthash=hashlib.sha256((l>0).tobytes()).hexdigest()
        assert key not in oldkeys|newkeys and supporthash not in oldsha|newsha
        assert a['passed'] and not any(a['boundary_flags'].values())
        assert t.shape==(16,32,32) and np.isfinite(t).all() and t.min()>=0 and t.max()<=1.1
        newkeys.add(key);newsha.add(supporthash)
        c.append(constant(t,l));h.append(t);labs.append(l)
        meta.append(dict(row,group_key=key,support_sha256=supporthash))
    truth=np.array([c,h]);labels=np.array(labs)
    kernel=Path('F:/2026.3/代码/G.mat');assert sha(kernel)==audit['identity']['kernel_sha256']
    with h5py.File(kernel) as f:g=np.asarray(f['G'],float).T*1000
    with threadpool_limits(2):inputs=(truth.reshape(400,-1).astype(float)@g.T).astype(np.float32).reshape(2,200,1,32,32)
    massdelta=[]
    for j,l in enumerate(labels):
        for k in np.unique(l[l>0]):
            m=l==k;massdelta.append(abs(truth[0,j][m].sum(dtype=float)-truth[1,j][m].sum(dtype=float))/truth[1,j][m].sum(dtype=float))
    assert max(massdelta)<1e-6
    sys.path.insert(0,str(CODE));from train_formal_20k import make_unet
    cm=json.loads((HERE/'constant_train10000_seed20260727/metadata.json').read_text(encoding='utf8'))
    assert sha(CODE/'train_formal_20k.py')==cm['resume_contract']['training_sources']['train_formal_20k.py']
    torch.set_num_threads(2);pred=np.empty((2,2,200,16,32,32),np.float32)
    for ai,w in enumerate(weights):
        net=make_unet().eval();net.load_state_dict(torch.load(w,map_location='cpu',weights_only=True))
        with torch.no_grad():
            for di in range(2):
                for j in range(0,200,16):pred[ai,di,j:j+16]=net(torch.from_numpy(inputs[di,j:j+16])).numpy()
        del net
    records=[];responses=[]
    for ai,arm in enumerate(['C','H']):
        for di,domain in enumerate(['C','H']):
            for j in range(200):
                p=pred[ai,di,j].astype(float);t=truth[di,j].astype(float);pm=p>.1;tm=t>0
                records.append(dict(arm=arm,domain=domain,id=j,family=meta[j]['family'],law=meta[j]['density_law_index'],
                    whole_rmse=float(np.sqrt(np.mean((p-t)**2))),dice=float(2*(pm&tm).sum()/(pm.sum()+tm.sum())),
                    mass_relative_error=float(abs(p.sum()-t.sum())/t.sum()),**recovery(p,t,labels[j])))
        for j in range(200):
            responses.append(dict(arm=arm,id=j,family=meta[j]['family'],law=meta[j]['density_law_index'],
                **recovery(pred[ai,1,j].astype(float)-pred[ai,0,j].astype(float),truth[1,j].astype(float)-truth[0,j].astype(float),labels[j])))
    summaries=[];effects=[];rng=np.random.default_rng(2026101601)
    for name,rr in [('response',[r for r in responses if r['law']!=0]),
                    ('H_nonconstant',[r for r in records if r['domain']=='H' and r['law']!=0]),
                    ('H_all',[r for r in records if r['domain']=='H']),('C_all',[r for r in records if r['domain']=='C'])]:
        metrics=['centered_rmse'] if name=='response' else ['whole_rmse','dice','mass_relative_error','centered_rmse','zone_mean_rmse']
        for arm in ['C','H']:
            z=[r for r in rr if r['arm']==arm]
            summaries.append(dict(target=name,arm=arm,n=len(z),**{k:float(np.mean([r[k] for r in z])) for k in metrics},
                positive_skill=sum(r['fluctuation_skill'] is not None and r['fluctuation_skill']>1e-8 for r in z)))
        a={r['id']:r for r in rr if r['arm']=='C'};b={r['id']:r for r in rr if r['arm']=='H'};ix=sorted(a);assert set(a)==set(b)
        draws=rng.integers(0,len(ix),size=(5000,len(ix)))
        for key in metrics:
            diff=np.array([a[i][key]-b[i][key] for i in ix])*(-1 if key=='dice' else 1)
            effects.append(dict(target=name,metric=key,gain_H=float(diff.mean()),pointwise95=np.quantile(diff[draws].mean(1),[.025,.975]).tolist()))
    out.mkdir()
    for filename,rr in [('per_case.csv',records),('response_per_case.csv',responses)]:
        with (out/filename).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rr[0]));writer.writeheader();writer.writerows(rr)
    np.savez_compressed(out/'arrays.npz',truth=truth,labels=labels,input=inputs,prediction=pred)
    (out/'new_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
    report=dict(summary=summaries,effects=effects,protocol_sha256=sha(protocol),script_sha256=sha(Path(__file__)),
        weights_sha256=[sha(w) for w in weights],max_body_mass_relative_error=float(max(massdelta)),
        old_geometry_overlap=0,old_support_hash_overlap=0,n_base_geometries=200,
        scope='Frozen same-generator new ID cases, not new training seeds or OOD validation')
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
