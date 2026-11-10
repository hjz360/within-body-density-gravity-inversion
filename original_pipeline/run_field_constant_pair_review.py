"""One fixed C/H field inference review; reuse frozen pipeline, no training."""
import os
import sys
import json
import hashlib
import time
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parent
CODE=Path('F:/2026.3/代码')
HROOT=Path('G:/2026.3/代码/paper_results/geology_v4_20260922')
OUT=BASE/'field_constant_H_pair_20261017_r1'


def check_model(model):
    a=np.asarray(model)
    if a.shape!=(16,36,43) or not np.isfinite(a).all() or np.any(a<0):
        raise ValueError('Expected finite nonnegative full field volume')
    return a


def section_slices(model):
    a=check_model(model)
    return a[:,15,:],a[:,:,20]


def shared_vmax(models):
    v=max(float(check_model(m).max()) for m in models)
    if v<=0:raise ValueError('Positive common density scale required')
    return v


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def run():
    started=time.monotonic()
    if OUT.exists():raise FileExistsError('Preserve existing inference attempt')
    OUT.mkdir()
    os.environ['MPLBACKEND']='Agg';os.environ['MPLCONFIGDIR']=str(OUT/'mpl_cache')
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
    sys.path.insert(0,str(CODE))
    import torch
    import matplotlib.pyplot as plt
    from threadpoolctl import threadpool_limits
    from eval_field_20k import load_field_windows,load_full_observed_gravity
    from field_evaluation import extract_windows,field_window_starts
    from field_observation_chain_stage_a import load_frozen_kriging_artifact
    from evaluate_c0c1_paired_v2_field import _infer_full_model,_load_operators,station_forward_prediction,STAGE_A
    torch.set_num_threads(2)
    old=HROOT/'vms_v2_dataset_10600/field_review_20260926'
    old_id=read(old/'manifest.json')['inputs']
    weights=[BASE/'constant_train10000_seed20260727/epoch_120_weights.pth',HROOT/'train10000_seed20260727/epoch_120_weights.pth']
    expected=['f816526e22d835f9779eadb6e17684dfbba469e3433a40534e9904bcaa69688f','1831249cd9aa01b698dd2260c0fca605d2d0cd7e37eca677a06b82c4d190212f']
    models_id=['geology_v4_constant_body_10000','geology_v4_balanced_10000']
    source={}
    def track(path,digest=None):
        value=sha(path)
        if digest is not None and value!=digest:raise ValueError('Source identity mismatch: '+str(path))
        source[str(path)]=value
    for i,w in enumerate(weights):
        meta=read(w.parent/'metadata.json')
        if meta['status']!='complete' or meta['completed_epochs']!=120 or meta['config']['seed']!=20260727 or meta['config']['model_id']!=models_id[i]:raise ValueError('Wrong fixed weight')
        track(w,expected[i]);track(w.parent/'metadata.json')
        if meta['artifact_sha256']['primary_weights']!=expected[i]:raise ValueError('Weight metadata disagrees')
    operators_path=STAGE_A/'operators/field_forward_operators.h5'
    kriging_path=STAGE_A/'kriging/frozen_field_kriging.mat'
    for path in [operators_path,kriging_path,*sorted((CODE/'Field_data/real').glob('data_slice*.mat'))]:track(path,old_id[str(path)])
    for path in [CODE/'Field_data/real/data.mat',CODE/'Field_data/real/pointdata.mat',old/'arrays.npz',old/'manifest.json',BASE.parent/'publication_comparison_v2/AA_reference_polyline.csv',BASE.parent/'publication_comparison_v2/BB_reference_polyline.csv',Path(__file__),BASE/'常值与非均匀网络实测对照_固定方案_20261017.md',BASE/'论文中文工作稿_v0.36_建模实证贡献版.md',BASE/'补充说明_v0.36.md']:
        track(path)
    for name in ['eval_field_20k.py','field_evaluation.py','evaluate_c0c1_paired_v2_field.py','field_observation_chain_stage_a.py','train_formal_20k.py']:track(CODE/name)
    write(OUT/'source_identity.json',source)
    windows=load_field_windows();full=load_full_observed_gravity()
    if windows.shape!=(60,1,32,32) or not np.isfinite(windows).all():raise ValueError('Invalid shared field input')
    np.testing.assert_allclose(extract_windows(full[None],field_window_starts()),windows,rtol=1e-5,atol=1e-5)
    archive=np.load(old/'arrays.npz',allow_pickle=False)
    np.testing.assert_array_equal(windows,archive['input_windows'])
    field=load_frozen_kriging_artifact(kriging_path)
    observed=np.asarray(field['station_gravity'],float).reshape(-1);xy=np.asarray(field['station_xy'])
    if len(observed)!=190:raise ValueError('Raw station count changed')
    np.testing.assert_array_equal(observed,archive['observed']);np.testing.assert_array_equal(xy,archive['station_xy'])
    operators=_load_operators(operators_path)
    models=[];rows=[];predictions={}
    with threadpool_limits(2):
        for arm,w in zip(['C','H'],weights):
            density=_infer_full_model(w,windows,torch.device('cpu'))
            models.append(check_model(density/1000))
            for branch,g in operators.items():
                p=station_forward_prediction(density,g);r=p-observed
                predictions[arm+'_'+branch]=p
                rows.append(dict(arm=arm,branch=branch,rmse_mgal=float(np.sqrt(np.mean(r*r))),relative_l2=float(np.linalg.norm(r)/np.linalg.norm(observed)),mean_residual_mgal=float(r.mean())))
            print('Completed fixed CPU inference:',arm,flush=True)
    anchor=float(np.max(np.abs(models[1]-archive['models_g_cm3'][0])))
    np.testing.assert_allclose(models[1],archive['models_g_cm3'][0],rtol=1e-4,atol=1e-6)
    for branch in operators:np.testing.assert_allclose(predictions['H_'+branch],archive['0_'+branch],rtol=1e-5,atol=1e-8)
    np.savez_compressed(OUT/'arrays.npz',C=models[0],H=models[1],input_windows=windows,observed=observed,station_xy=xy,**predictions)
    paths=[np.loadtxt(BASE.parent/('publication_comparison_v2/'+n+'_reference_polyline.csv'),delimiter=',',skiprows=1)[:,2:] for n in ['AA','BB']]
    for p in paths:np.testing.assert_array_equal(p[0],p[-1])
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei'],'axes.unicode_minus':False,'pdf.fonttype':42})
    vmax=shared_vmax(models);cmap=plt.get_cmap('turbo').copy();cmap.set_bad('white')
    for masked,stem in [(False,'01_常值与非均匀_完整密度范围'),(True,'02_常值与非均匀_统一白底显示')]:
        fig,axes=plt.subplots(2,2,figsize=(12,7.1),layout='constrained')
        for j,(arm,m) in enumerate(zip(['C：逐体常值训练库','H：体内非均匀训练库'],models)):
            for i,sl in enumerate(section_slices(m)):
                ax=axes[i,j];lo,hi=[(-2725,-575),(-1175,625)][i]
                display=np.ma.masked_less(sl,.05) if masked else sl
                im=ax.imshow(display,extent=[lo,hi,800,0],vmin=0,vmax=vmax,cmap=cmap,aspect='equal',interpolation='nearest')
                ax.plot(*paths[i].T,color='#854321',lw=1.35,zorder=5)
                ax.set_xticks(np.arange(lo,hi+1,50),minor=True);ax.set_yticks(np.arange(0,801,50),minor=True)
                ax.grid(which='minor',color='.75',lw=.3,alpha=.65);ax.tick_params(which='minor',length=0)
                ax.set(xlabel=('东向' if i==0 else '北向')+' / m',ylabel='深度 / m',title=f'({chr(97+i*2+j)}) {arm}')
        fig.colorbar(im,ax=axes.ravel().tolist(),label='密度差 / (g/cm³)',shrink=.8)
        fig.suptitle('圣尼古拉斯矿床：配对训练网络的实测密度剖面对照',fontsize=15)
        footer='上排 A–A′：北向−400 m；下排 B–B′：东向−1700 m。固定种子20260727，第120轮，同输入与色标。'
        footer+='\n棕线：文献地质参照，非密度真值。'+('白色表示密度差＜0.05 g/cm³，仅作用于显示。' if masked else '完整模型显示，不隐藏低密度。')
        fig.supxlabel(footer,fontsize=10)
        for ext in ['png','pdf']:fig.savefig(OUT/(stem+'.'+ext),dpi=220)
        plt.close(fig)
    for path,digest in source.items():
        if sha(path)!=digest:raise ValueError('Source changed during review: '+path)
    write(OUT/'metrics.json',dict(scope='Data-fit audit only; no field density ground truth',rows=rows,shared_density_vmax_g_cm3=vmax,H_archive_max_abs_difference_g_cm3=anchor,station_count=190,seed=20260727,epoch=120))
    artifacts={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}
    write(OUT/'verification.json',dict(passed=True,elapsed_seconds=time.monotonic()-started,input_exactly_matches_archive=True,H_archive_anchor_pass=True,source_files_unchanged=True,source_count=len(source),no_training=True,no_field_selection=True,torch=torch.__version__,numpy=np.__version__,artifact_sha256=artifacts))
    print(json.dumps(dict(output=str(OUT),rows=rows,anchor=anchor,elapsed_seconds=time.monotonic()-started),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':run()
