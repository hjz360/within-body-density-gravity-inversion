"""Post-hoc author-side zero-reference analysis and fixed field seed sensitivity."""
from pathlib import Path
import csv, hashlib, json, os, sys, math, argparse
import numpy as np

BASE=Path(__file__).resolve().parent
SEEDS=(20260727,20260728,20260729)
ROOT=Path('G:/2026.3/代码/paper_results/geology_v4_20260922')
CODE=Path('F:/2026.3/代码')
PROTOCOL=BASE/'v045_有限修订执行方案_20261020.md'
ZERO=BASE/'zero_reference_20261020_r1'
FIELD=BASE/'field_seed_sensitivity_20261020_r1'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(p,data):
    Path(p).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def summarize_response(rows,expected_n=160):
    groups={a:{} for a in ('C','H')}
    for r in rows:
        if int(r['law'])==0:continue
        arm=r['arm'];i=int(r['id'])
        if arm not in groups or i in groups[arm]:raise ValueError('Unknown arm or duplicate case')
        t,e,s=map(float,(r['truth_fluctuation_rms'],r['centered_rmse'],r['fluctuation_skill']))
        if not all(map(math.isfinite,(t,e,s))) or t<=0 or e<0:raise ValueError('Invalid nonconstant metric')
        if abs(s-(1-(e/t)**2))>2e-10:raise ValueError('Stored skill disagrees with error and reference')
        groups[arm][i]=(t,e,s)
    if set(groups['C'])!=set(groups['H']) or len(groups['C'])!=expected_n:raise ValueError('Wrong paired case population')
    for i in groups['C']:
        if abs(groups['C'][i][0]-groups['H'][i][0])>1e-12:raise ValueError('Reference mismatch between networks')
    z=float(np.mean([v[0] for v in groups['C'].values()]))
    out=dict(n=expected_n,zero_response_mean_rmse=z)
    for a,g in groups.items():
        vals=np.asarray(list(g.values()),float);err=float(vals[:,1].mean())
        out[a]=dict(mean_rmse=err,relative_mean_rmse_percent=100*(err/z-1),
                    mean_skill=float(vals[:,2].mean()),median_skill=float(np.median(vals[:,2])),
                    positive_skill=int(np.count_nonzero(vals[:,2]>0)))
    return out


def roi_summary(c,h):
    for a in (c,h):
        if a.shape!=(16,36,43) or not np.isfinite(a).all() or np.any(a<0):raise ValueError('Invalid field volume')
    north=-1175+(np.arange(36)+.5)*50
    depth=(np.arange(16)+.5)*50
    mask=np.broadcast_to(((depth[:,None,None]>=400)&(depth[:,None,None]<=800)&
                          (north[None,:,None]>=-900)&(north[None,:,None]<-700)),c.shape)
    cv,hv=c[mask],h[mask]
    return dict(voxel_count=int(mask.sum()),mean_C_g_cm3=float(cv.mean()),mean_H_g_cm3=float(hv.mean()),
                mean_H_minus_C_g_cm3=float((hv-cv).mean()),
                integral_C_g_cm3_m3=float(cv.sum()*50**3),integral_H_g_cm3_m3=float(hv.sum()*50**3),
                threshold_counts={f'{t:.2f}':dict(C=int(np.count_nonzero(cv>=t)),H=int(np.count_nonzero(hv>=t))) for t in (.05,.1,.2)})


def classify_directions(values):
    if len(values)!=3 or not np.isfinite(values).all():raise ValueError('Exactly three finite paired directions required')
    n=sum(v>0 for v in values)
    return dict(positive_pairs=n,label={3:'当前三对方向一致',2:'存在种子差异',1:'种子敏感性示例',0:'未出现预设区域的相对增强'}[n])


def weight_path(seed,arm):
    if seed==20260727:
        return (BASE/'constant_train10000_seed20260727' if arm=='C' else ROOT/'train10000_seed20260727')/'epoch_120_weights.pth'
    return BASE/f'paired_repeats_20261016/{arm}_seed{seed}/epoch_120_weights.pth'


def run_zero():
    if ZERO.exists():raise FileExistsError(ZERO)
    ZERO.mkdir()
    sources={str(PROTOCOL):sha(PROTOCOL),str(Path(__file__)):sha(__file__)}
    path=BASE/'frozen_followup_200_seed20260727/arrays.npz'
    sources[str(path)]=sha(path)
    with np.load(path,allow_pickle=False) as z:truth=z['truth'];labels=z['labels']
    if truth.shape!=(2,200,16,32,32) or labels.shape!=(200,16,32,32):raise ValueError('Wrong target arrays')
    groups=[];case_records=[];maxdiff=0.;reference=None
    for seed in SEEDS:
        cp=BASE/f'paired_repeats_20261016/evaluations/seed{seed}/followup200_responses.csv'
        pp=cp.with_name('followup200_predictions.npz')
        for p in (cp,pp):sources[str(p)]=sha(p)
        with cp.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        s=summarize_response(rows);s['seed']=seed
        key={(r['arm'],int(r['id'])):r for r in rows if int(r['law'])!=0}
        current={i:float(r['truth_fluctuation_rms']) for (a,i),r in key.items() if a=='C'}
        if reference is not None and current!=reference:raise ValueError('References differ between seeds')
        reference=current
        with np.load(pp,allow_pickle=False) as z:pred=z['prediction'];ids=z['ids']
        np.testing.assert_array_equal(ids,np.arange(200))
        if pred.shape!=(2,2,200,16,32,32):raise ValueError('Wrong prediction dimensions')
        # Independent array path: per-body vectors, not CSV skill formula aggregation.
        for i in range(200):
            lab=labels[i];inside=lab>0
            vec=(truth[1,i].astype(float)-truth[0,i].astype(float))
            y=vec.copy()
            for k in np.unique(lab[inside]):y[lab==k]-=vec[lab==k].mean()
            tv=y[inside];t=float(np.linalg.norm(tv)/math.sqrt(tv.size))
            if i not in current:
                if t>1e-7:raise ValueError('Unreported nonconstant target')
                continue
            for ai,a in enumerate(('C','H')):
                pv=pred[ai,1,i].astype(float)-pred[ai,0,i].astype(float)
                q=pv.copy()
                for k in np.unique(lab[inside]):q[lab==k]-=pv[lab==k].mean()
                err=q[inside]-tv;energy=float(err@err);denom=float(tv@tv)
                e=math.sqrt(energy/tv.size);skill=1-energy/denom
                r=key[(a,i)]
                diffs=[abs(t-float(r['truth_fluctuation_rms'])),abs(e-float(r['centered_rmse'])),abs(skill-float(r['fluctuation_skill']))]
                maxdiff=max(maxdiff,*diffs)
                if max(diffs)>2e-10:raise ValueError(('Array verification mismatch',seed,a,i,diffs))
                case_records.append(dict(seed=seed,arm=a,id=i,zero_response_rmse=t,response_rmse=e,skill=skill))
        groups.append(s)
    history=[]
    for seed in SEEDS:
        for a in ('C','H'):
            w=weight_path(seed,a);m=w.parent/'metadata.json';hp=w.parent/'history.npz'
            for p in (m,hp):sources[str(p)]=sha(p)
            meta=read(m)
            with np.load(hp,allow_pickle=False) as z:
                payload={k:z[k].tolist() for k in z.files}
            history.append(dict(seed=seed,arm=a,best_epoch=meta['best_epoch'],
                                best_validation_total=meta['best_validation_total'],
                                final_validation_total=meta['final_validation']['total'],
                                best_weights_exist=(w.parent/'best_validation_weights.pth').is_file(),
                                history_keys=list(payload),history_epochs=payload))
    for p,h in sources.items():
        if sha(p)!=h:raise ValueError('Source changed: '+p)
    write(ZERO/'summary.json',dict(groups=groups,post_hoc=True,author_side=True,not_external_review=True,
                                   weighting='Equal weight per nonconstant geometry, n=160; skill is mean of per-case energy ratios',
                                   array_verification=dict(passed=True,network_case_records=len(case_records),max_absolute_difference=maxdiff),
                                   source_sha256=sources))
    write(ZERO/'validation_history.json',dict(rows=history,no_best_checkpoint_inference=True))
    with (ZERO/'per_case.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(case_records[0]));w.writeheader();w.writerows(case_records)
    print(json.dumps(dict(groups=groups,array_maxdiff=maxdiff,history=[{k:v for k,v in r.items() if k!='history_epochs'} for r in history]),ensure_ascii=False,indent=2))


def render_field(models):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei'],'axes.unicode_minus':False,'pdf.fonttype':42})
    paths=[np.loadtxt(BASE.parent/f'publication_comparison_v2/{a}_reference_polyline.csv',delimiter=',',skiprows=1)[:,2:] for a in ('AA','BB')]
    vmax=max(float(m.max()) for m in models.values());cmap=plt.get_cmap('turbo').copy();cmap.set_bad('white')
    for section,extent in [('AA',[-2725,-575,800,0]),('BB',[-1175,625,800,0])]:
        for masked in (False,True):
            fig,axes=plt.subplots(3,2,figsize=(12,9.7),layout='constrained')
            for i,seed in enumerate(SEEDS):
                for j,a in enumerate(('C','H')):
                    m=models[(seed,a)];sl=m[:,15,:] if section=='AA' else m[:,:,20]
                    ax=axes[i,j];im=ax.imshow(np.ma.masked_less(sl,.05) if masked else sl,extent=extent,vmin=0,vmax=vmax,cmap=cmap,interpolation='nearest',aspect='equal')
                    path=paths[0 if section=='AA' else 1]
                    ax.plot(*path.T,color='#854321',lw=1)
                    if section=='BB':ax.axvspan(-900,-700,color='gray',alpha=.10);ax.axhline(400,color='gray',lw=.6,ls='--')
                    ax.set(title=f'{seed} · {a}',xlabel=('东向' if section=='AA' else '北向')+' / m',ylabel='深度 / m')
                    ax.tick_params(labelsize=9)
            fig.colorbar(im,ax=axes.ravel().tolist(),label='密度差 / (g/cm³)',shrink=.85)
            fig.suptitle(f'三对固定训练网络的San Nicolás {section}剖面',fontsize=15)
            fig.supxlabel('各面板同色标；第120轮、同60窗口、同重叠平均；棕线为文献参照，非密度真值。\n'+('白色仅表示＜0.05 g/cm³。' if masked else '完整密度范围，不隐藏低幅度。'),fontsize=10)
            stem=f'{section}_'+('统一白底' if masked else '完整范围')
            for ext in ('png','pdf'):fig.savefig(FIELD/f'{stem}.{ext}',dpi=200)
            plt.close(fig)


def run_field():
    if FIELD.exists():raise FileExistsError(FIELD)
    FIELD.mkdir()
    os.environ['MPLBACKEND']='Agg';os.environ['MPLCONFIGDIR']=str(FIELD/'mpl_cache')
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
    sys.path.insert(0,str(CODE))
    import torch
    from threadpoolctl import threadpool_limits
    from eval_field_20k import load_field_windows,load_full_observed_gravity
    from field_evaluation import extract_windows,field_window_starts
    from field_observation_chain_stage_a import load_frozen_kriging_artifact
    from evaluate_c0c1_paired_v2_field import _infer_full_model,_load_operators,station_forward_prediction,STAGE_A
    torch.set_num_threads(2)
    previous=BASE/'field_constant_H_pair_20261017_r1'
    identity=read(previous/'source_identity.json')
    sources={str(PROTOCOL):sha(PROTOCOL),str(Path(__file__)):sha(__file__)}
    for p,h in identity.items():
        if Path(p).is_file() and (str(CODE) in p or 'epoch_120_weights' in p):
            if sha(p)!=h:raise ValueError('Legacy source mismatch: '+p)
            sources[p]=h
    for p in (previous/'arrays.npz',previous/'metrics.json',previous/'source_identity.json'):
        sources[str(p)]=sha(p)
    windows=load_field_windows();full=load_full_observed_gravity()
    if windows.shape!=(60,1,32,32) or not np.isfinite(windows).all():raise ValueError('Invalid input')
    np.testing.assert_allclose(extract_windows(full[None],field_window_starts()),windows,rtol=1e-5,atol=1e-5)
    with np.load(previous/'arrays.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(windows,z['input_windows'])
        models={(20260727,a):z[a] for a in ('C','H')};observed=z['observed'];xy=z['station_xy']
    artifact=load_frozen_kriging_artifact(STAGE_A/'kriging/frozen_field_kriging.mat')
    np.testing.assert_array_equal(observed,np.asarray(artifact['station_gravity']).reshape(-1))
    np.testing.assert_array_equal(xy,artifact['station_xy'])
    if observed.size!=190:raise ValueError('Station count changed')
    operators=_load_operators(STAGE_A/'operators/field_forward_operators.h5')
    for seed in SEEDS:
        for a in ('C','H'):
            w=weight_path(seed,a);meta=read(w.parent/'metadata.json');digest=sha(w)
            if meta['status']!='complete' or meta['completed_epochs']!=120 or meta['config']['seed']!=seed:raise ValueError('Wrong completed weight')
            if meta['artifact_sha256']['primary_weights']!=digest:raise ValueError('Weight hash mismatch')
            expected='geology_v4_constant_body_10000' if a=='C' else 'geology_v4_balanced_10000'
            if meta['config']['model_id']!=expected:raise ValueError('Wrong training arm')
            for p in (w,w.parent/'metadata.json'):sources[str(p)]=sha(p)
    rows=[];regions=[];predictions={}
    with threadpool_limits(2):
        for seed in SEEDS:
            for a in ('C','H'):
                if seed!=20260727:
                    models[(seed,a)]=_infer_full_model(weight_path(seed,a),windows,torch.device('cpu'))/1000
                    print(f'Fixed field inference complete: {seed} {a}',flush=True)
                m=models[(seed,a)]
                roi_summary(m,m)
                for branch,kernel in operators.items():
                    p=station_forward_prediction(m*1000,kernel);res=p-observed
                    predictions[f'{seed}_{a}_{branch}']=p
                    rows.append(dict(seed=seed,arm=a,branch=branch,rmse_mgal=float(np.sqrt(np.mean(res**2)))))
            regions.append(dict(seed=seed,**roi_summary(models[(seed,'C')],models[(seed,'H')])))
    # Original anchor scoring must remain unchanged.
    old=read(previous/'metrics.json')['rows']
    for r in rows[:4]:
        anchor=next(v for v in old if v['arm']==r['arm'] and v['branch']==r['branch'])
        if abs(r['rmse_mgal']-anchor['rmse_mgal'])>1e-12:raise ValueError('Original field scoring changed')
    np.savez_compressed(FIELD/'arrays.npz',**{f'{s}_{a}':m for (s,a),m in models.items()},input_windows=windows,observed=observed,station_xy=xy,**predictions)
    classification=classify_directions([r['mean_H_minus_C_g_cm3'] for r in regions])
    render_field(models)
    for p,h in sources.items():
        if sha(p)!=h:raise ValueError('Source changed: '+p)
    write(FIELD/'summary.json',dict(regions=regions,classification=classification,station_fit=rows,
          station_count=190,new_inference_networks=4,reused_archived_networks=2,epoch=120,
          post_hoc=True,author_side=True,not_external_review=True,no_training=True,
          interpretation='Regional enhancement, not proof of a separated body or true field density',
          source_sha256=sources,artifact_sha256={p.name:sha(p) for p in FIELD.iterdir() if p.is_file()}))
    print(json.dumps(dict(regions=regions,classification=classification,station_fit=rows),ensure_ascii=False,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['zero','field']);args=ap.parse_args()
    run_zero() if args.mode=='zero' else run_field()
