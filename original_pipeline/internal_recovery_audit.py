"""Read-only reanalysis of saved predictions and fixed training examples.

No training, no inference, no reserved-test access. Outputs use a fresh G: directory.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MPLBACKEND', 'Agg')
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import ndimage

BASE = Path('G:/2026.3/代码/paper_results')
ROOT = BASE / 'geology_v4_20260922/structure_candidate_dataset_10600_20260927_r1'
OUT = ROOT / 'internal_recovery_scale_20260927_r1'
ARM_NAMES = ['旧版混合库', 'VMS三类库', '本轮六类库']


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))


def write(p, x):
    Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def core_labels(labels, iterations):
    labels = np.asarray(labels)
    if labels.ndim != 3 or not np.isfinite(labels).all() or np.any(labels < 0) or np.any(labels != labels.astype(int)):
        raise ValueError('invalid region labels')
    if type(iterations) is not int or iterations < 0: raise ValueError('invalid erosion')
    result = np.zeros_like(labels)
    for k in np.unique(labels[labels > 0]):
        mask = labels == k
        if iterations: mask = ndimage.binary_erosion(mask, iterations=iterations, border_value=0)
        result[mask] = k
    return result


def recovery(pred, truth, labels):
    p = np.asarray(pred, float); t = np.asarray(truth, float); lab = core_labels(labels, 0)
    if p.shape != t.shape or p.shape != lab.shape or not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError('invalid prediction or truth')
    mask = lab > 0; n = int(mask.sum())
    keys = ['rmse', 'zone_mean_rmse', 'centered_rmse', 'truth_fluctuation_rms',
            'prediction_fluctuation_rms', 'fluctuation_skill', 'amplitude_ratio', 'correlation']
    if not n: return dict(n=0, **{k: None for k in keys})
    sse = mean_sse = centered_sse = tt = pp = pt = 0.
    for k in np.unique(lab[mask]):
        a = p[lab == k]; b = t[lab == k]
        ac = a - a.mean(); bc = b - b.mean()
        sse += np.sum((a-b)**2); mean_sse += len(a)*(a.mean()-b.mean())**2
        centered_sse += np.sum((ac-bc)**2)
        tt += bc @ bc; pp += ac @ ac; pt += ac @ bc
    np.testing.assert_allclose(sse, mean_sse + centered_sse, rtol=1e-10, atol=1e-12)
    # Float32 constant labels can differ by roundoff: do not invent a correlation.
    tol = n * (1e-7 * max(1., float(np.abs(t[mask]).max())))**2
    return dict(n=n, rmse=float(np.sqrt(sse/n)), zone_mean_rmse=float(np.sqrt(mean_sse/n)),
                centered_rmse=float(np.sqrt(centered_sse/n)), truth_fluctuation_rms=float(np.sqrt(tt/n)),
                prediction_fluctuation_rms=float(np.sqrt(pp/n)),
                fluctuation_skill=float(1-centered_sse/tt) if tt > tol else None,
                amplitude_ratio=float(np.sqrt(pp/tt)) if tt > tol else None,
                correlation=float(np.clip(pt/np.sqrt(pp*tt), -1, 1)) if min(tt, pp) > tol else None)


def constant_projection(density, occupancy):
    r = np.asarray(density, float); o = np.asarray(occupancy, float)
    if r.shape != o.shape or r.ndim != 4 or not np.isfinite(r).all() or not np.isfinite(o).all():
        raise ValueError('invalid density/occupancy shape or values')
    if np.any(r < -1e-10) or np.any(o < 0) or np.any(o.sum(0) > 1+1e-6) or np.any((o == 0) & (r > 1e-10)):
        raise ValueError('invalid density/occupancy support')
    vol = o.sum(axis=(1, 2, 3)); mass = r.sum(axis=(1, 2, 3))
    if vol.sum() <= 0: raise ValueError('empty occupancy')
    mean = np.divide(mass, vol, out=np.zeros_like(mass), where=vol > 0)
    result = o * mean[:, None, None, None]
    np.testing.assert_allclose(result.sum(axis=(1, 2, 3)), mass, rtol=1e-12, atol=1e-12)
    return result


def regions(lab):
    c1 = core_labels(lab, 1); c2 = core_labels(lab, 2)
    return [('support', lab), ('core1', c1), ('core2', c2), ('boundary1', np.where(c1 > 0, 0, lab))]


def table(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def aggregate(rows, groupkeys):
    metrics = ['rmse', 'zone_mean_rmse', 'centered_rmse', 'truth_fluctuation_rms',
               'prediction_fluctuation_rms', 'fluctuation_skill', 'amplitude_ratio', 'correlation']
    output = []
    for key in sorted({tuple(r[k] for k in groupkeys) for r in rows}):
        subset = [r for r in rows if tuple(r[k] for k in groupkeys) == key]
        entry = dict(zip(groupkeys, key)); entry['cases'] = len(subset)
        entry['nonempty_cases'] = sum(r['n'] > 0 for r in subset)
        for m in metrics:
            vals = [r[m] for r in subset if r[m] is not None]
            entry[m] = float(np.mean(vals)) if vals else None
            entry[m+'_valid_count'] = len(vals)
        output.append(entry)
    return output


def run():
    if OUT.exists(): raise FileExistsError('preserve existing audit: '+str(OUT))
    inputs = {}; checks = []
    def track(p):
        p = Path(p); inputs[str(p)] = sha(p); return p
    def verified(directory, file, manifest_name='manifest.json', section='outputs'):
        manifest = read(track(directory / manifest_name))
        p = track(directory / file)
        if inputs[str(p)] != manifest[section][file]: raise ValueError('source digest changed: '+str(p))
        checks.append('historical digest verified: '+str(p)); return p
    theory = BASE / 'geology_model_prototype_20260920/independent_theory_v1'
    review = ROOT / 'theory12_review_20260927_r1'
    archive = np.load(verified(review, 'evaluation_arrays.npz'))
    hashes = read(track(theory / 'array_hashes.json'))
    for name in ['density.npy', 'body_labels.npy']:
        p = track(theory/name)
        if inputs[str(p)] != hashes[name]: raise ValueError('theory array changed')
    t = np.load(theory/'density.npy'); labels = np.load(theory/'body_labels.npy')
    np.testing.assert_array_equal(t, archive['truth'])
    meta = read(track(theory/'metadata.json'))
    if t.shape != (12,16,32,32) or archive['prediction'].shape != (3,12,16,32,32): raise ValueError('theory shape')
    rows = []
    for a, name in enumerate(ARM_NAMES):
        for i in range(12):
            for region, lab in regions(labels[i]):
                rows.append(dict(arm=name, index=i, name=meta[i]['name'], law=meta[i]['law'], region=region,
                                 **recovery(archive['prediction'][a,i], t[i], lab)))
    theory_summary = aggregate(rows, ['law','region','arm'])
    # Old experiment: only the saved first ID/OOD example, not the full test set.
    oldrows = []; old_truth = {}; old_observation = {}; fixed_ids = {}
    follow = BASE/'label_consistency_followup_20260914'
    for seed in [20260727,20260728,20260729]:
        root = follow if seed == 20260727 else follow/'multiseed'/f'seed_{seed}'
        ev = root/'selfconsistent_evaluation'
        z = np.load(track(ev/'fixed_examples.npz'))
        records = read(track(ev/'per_sample_metrics.json'))
        for model in ['C0_shared','C0_selfconsistent','C1_shared']:
            for domain in ['C0','C1']:
                for split in ['id_test','ood_layered']:
                    prefix = f'{model}_{domain}_{split}'
                    p = z[prefix+'_prediction']; truth = z[prefix+'_truth']; obs = z[prefix+'_observed']
                    rec = next(r for r in records if r['model']==model and r['domain']==domain and r['split']==split)
                    np.testing.assert_allclose(np.sqrt(np.mean((p-truth)**2)), rec['density_rmse'], rtol=2e-5, atol=1e-8)
                    key = (domain,split)
                    if key in old_truth:
                        np.testing.assert_array_equal(truth,old_truth[key]); np.testing.assert_array_equal(obs,old_observation[key])
                        if rec['base_model_id']!=fixed_ids[key]: raise ValueError('fixed example ID mismatch')
                    old_truth[key]=truth; old_observation[key]=obs; fixed_ids[key]=rec['base_model_id']
                    # Geological zone labels were not archived here. Union support is explicit.
                    for region, lab in regions((truth>0).astype(np.uint8)):
                        oldrows.append(dict(seed=seed, arm=model, domain=domain, split=split,
                                            base_model_id=rec['base_model_id'], region=region, **recovery(p,truth,lab)))
    checks.append('old saved examples match archived whole-grid RMSE and share truth/observations across seeds and arms; no historical NPZ digest available')
    old_summary = aggregate(oldrows,['domain','split','region','arm'])
    # Geometry/occupancy: reuse the preselected 60 training examples without regeneration.
    zone = ROOT/'zone_boundary_audit_20260927_r1'
    z = np.load(verified(zone,'selected_zone_contributions.npz'))
    selection = read(verified(zone,'selection.json'))
    ids = z['indices'].astype(int)
    if len(ids)!=60 or set(ids)!=set(selection['ids']) or len(set(ids))!=60 or np.any(ids>=10000): raise ValueError('training selection changed')
    metadata = read(track(ROOT/'metadata.json'))
    import structure_candidate_sampling as sampler
    identity = read(track(ROOT/'identity.json'))
    for key, file in [('generator','structure_candidate_sampling.py'),('integration','structure_atlas_candidate.py'),('legacy_density','vms_v2.py')]:
        if sha(track(file))!=identity[key]: raise ValueError('generator changed: '+file)
    import h5py
    kernel_path = track('F:/2026.3/代码/G.mat')
    prior_manifest = read(review/'manifest.json')
    if sha(kernel_path)!=prior_manifest['inputs'][str(kernel_path)]: raise ValueError('forward kernel changed')
    with h5py.File(kernel_path) as h: G = np.asarray(h['G'],dtype=float).T*1000
    if G.shape!=(1024,16384): raise ValueError('kernel orientation')
    np.testing.assert_allclose(t.reshape(12,-1)@G.T, archive['input'].reshape(12,-1),rtol=1e-5,atol=1e-5)
    pairs=[]; scale=[]; const=[]
    for j,idx in enumerate(ids):
        rec = metadata[idx]
        if rec['split']!='train' or rec['pool_id']!=idx: raise ValueError('nontraining sample')
        c = sampler.sample('train',rec['family'],int(rec['group_id'].rsplit('-',1)[1]))
        if c['row']['group_id']!=rec['group_id']: raise ValueError('reconstruction ID')
        r = z['density'][j].astype(float); o = z['occupancy'][j].astype(float)
        cr = constant_projection(r,o); const.append(cr.sum(0))
        occ = o.sum(0); support = occ>0; full = occ>=.999; pure = np.max(o,axis=0)>=.999
        g = c['row']['geometry']; family=rec['family']
        scale.append(dict(index=int(idx),family=family,name=sampler.NAMES[family],
                          half_parameter_twice_m=2*g['half'],half_parameter_twice_cells=2*g['half']/50,
                          occupied_cells=int(support.sum()),full_cells=int(full.sum()),pure_zone_full_cells=int(pure.sum()),
                          full_cell_fraction=float(full.sum()/support.sum()),pure_full_fraction=float(pure.sum()/support.sum()),
                          core1_cells=int((core_labels(support.astype(int),1)>0).sum()),
                          partial_mass_fraction=float(r.sum(0)[support & ~full].sum()/r.sum())))
    variable=z['density'].sum(1).astype(float); constant=np.array(const)
    gv=variable.reshape(60,-1)@G.T; gc=constant.reshape(60,-1)@G.T
    for j,s in enumerate(scale):
        delta=gv[j]-gc[j]; peak=float(np.max(np.abs(gv[j])))
        pairs.append(dict(index=s['index'],family=s['family'],name=s['name'],
                          density_rms_difference=float(np.sqrt(np.mean((variable[j]-constant[j])**2))),
                          relative_mass_error=float(abs(variable[j].sum()-constant[j].sum())/variable[j].sum()),
                          forward_relative_l2=float(np.linalg.norm(delta)/np.linalg.norm(gv[j])),
                          forward_difference_rmse_mgal=float(np.sqrt(np.mean(delta**2))),
                          ratio_to_2pct_peak_noise_sigma=float(np.sqrt(np.mean(delta**2))/(.02*peak)),
                          ratio_to_4pct_peak_noise_sigma=float(np.sqrt(np.mean(delta**2))/(.04*peak))))
    scale_summary=[]
    for f in range(6):
        ss=[s for s in scale if s['family']==f]; ps=[s for s in pairs if s['family']==f]
        scale_summary.append(dict(family=f,name=sampler.NAMES[f],n=len(ss),
            parameter_2half_median_m=float(np.median([s['half_parameter_twice_m'] for s in ss])),
            parameter_2half_min_m=float(min(s['half_parameter_twice_m'] for s in ss)),
            parameter_2half_max_m=float(max(s['half_parameter_twice_m'] for s in ss)),
            no_pure_full_cells=sum(s['pure_zone_full_cells']==0 for s in ss),
            median_pure_full_fraction=float(np.median([s['pure_full_fraction'] for s in ss])),
            median_partial_mass_fraction=float(np.median([s['partial_mass_fraction'] for s in ss])),
            median_forward_relative_l2=float(np.median([s['forward_relative_l2'] for s in ps])),
            median_noise_ratio_2pct=float(np.median([s['ratio_to_2pct_peak_noise_sigma'] for s in ps]))))
    # All reads and validity checks above finish before a new output directory is made.
    OUT.mkdir(); figs=OUT/'figures'; figs.mkdir()
    os.environ['MPLCONFIGDIR']=str(OUT/'matplotlib_cache')
    write(OUT/'protocol.json',dict(scope='saved development predictions; no independent generalization claim',
          sealed_test_used=False,training=False,inference=False,erosion='six-neighbor 1/2 iterations per truth body; not exact Euclidean depth',
          old_scope='4 truth/domain cases, 2 geometry IDs, 3 seeds; union support, no geological zone labels',
          scale_scope='same 60 fixed training indices; 10 per family, not random or prevalence weighted',
          constant_reference='preserve each zone mass and occupancy; no network trained on this projection',
          noise_reference='difference RMSE / (0.02 or 0.04 times variable input peak); descriptive, not a detectability test',
          fluctuation_skill='1 - within-region centered error SSE / truth within-region fluctuation SSE; null for constant truth',
          oracle_reference='zero fluctuation after truth-region centering; uses truth masks/means, not a deployable inversion method'))
    for name,data in [('理论逐例内部指标',rows),('旧自洽固定个例指标',oldrows),('六十例尺度诊断',scale),('六十例分区常值正演差',pairs)]: table(OUT/(name+'.csv'),data)
    write(OUT/'summary.json',dict(theory=theory_summary,old_fixed_examples=old_summary,scale=scale_summary))
    np.savez_compressed(OUT/'paired_training_diagnostic.npz',indices=ids,variable=variable,constant=constant,gravity_variable=gv,gravity_constant=gc)
    make_figures(figs,rows,oldrows,scale_summary)
    lines=['# 内部密度恢复与50米网格尺度诊断','','## 范围与口径',
           '本轮仅重算已存预测和固定60训练样本；未训练、未重新网络推理、未解封600保留参数。',
           '12例理论模型已参与开发，不是盲测；旧自洽缓存只有2个几何ID×2种真值，三种子不是新增独立地质样本。',
           '理论逐体去均值；旧缓存无分区标签，只对整体真值支撑去均值。两者不得直接合并统计。',
           'RMSE²=分区均值误差²+分区内去均值误差²。所有指标使用真值区域，包括漏检体素。',
           '起伏技能=1−去均值误差平方和/真值起伏平方和。1为完美，0相当于不恢复区内起伏，负值更差；该参考使用真值区域，不是可部署反演基线。',
           '常值真值不计算技能或相关性，预测起伏RMS表示虚假起伏。支撑向内六邻域腐蚀1/2层仅为边界敏感性分析；空核心标NA并保留计数，不当零误差。',
           '', '## 理论模型：各体去均值的起伏恢复',
           '|真值|区域|网络|有效案例|平均去均值RMSE|平均起伏技能|', '|---|---|---|---:|---:|---:|']
    for r in theory_summary:
        if r['region'] not in ['support','core1']: continue
        val=lambda k: 'NA' if r[k] is None else f'{r[k]:.5f}'
        lines.append(f'|{r["law"]}|{r["region"]}|{r["arm"]}|{r["nonempty_cases"]}/{r["cases"]}|{val("centered_rmse")}|{val("fluctuation_skill")}|')
    lines += ['', '## 训练模型尺度与物理配对',
              '2×half是各生成器的几何参数尺度，不是统一的局部法向厚度：主体为中心面相关尺度，双体为中心垂向尺度，弯曲层为垂向厚度，倾斜薄体为法向厚度。不得直接把所有类别2×half混称同一种厚度。',
              '纯分区满格：某一分区占据率≥0.999；这是积分离散意义的格内条件，不是数学上远离界面的保证。',
              '|类型|样本数|无纯分区满格|纯满格比例中位数|2half范围m|正演相对差中位数|差值RMSE/2%峰值噪声σ|',
              '|---|---:|---:|---:|---|---:|---:|']
    for s in scale_summary:
        lines.append(f'|{s["name"]}|{s["n"]}|{s["no_pure_full_cells"]}|{s["median_pure_full_fraction"]:.3f}|{s["parameter_2half_min_m"]:.1f}–{s["parameter_2half_max_m"]:.1f}|{s["median_forward_relative_l2"]:.4%}|{s["median_noise_ratio_2pct"]:.3f}|')
    lines += ['', '分区常值投影保留原占据率和每区总质量；不是整格填满，也不是新版常值训练实验。正演使用已核验同一G，仅为内部一致性比较。',
              '差值与噪声σ的比值不是统计检验或可恢复性充分条件；多个测站的空间结构、噪声协方差和反演非唯一性仍影响判断。',
              '', '## 旧自洽固定个例',
              'CSV保留所有三种子和三网络，不隐藏C0_shared。每个缓存全域RMSE与历史逐例记录核对，真值与输入在三网络和三种子之间逐元素相同；没有历史缓存SHA，故不能声称历史哈希溯源完整。',
              '本节仅是固定个例补充，不能将其平均值称为1256个测试模型或三种子全测试集的内部恢复结论。',
              '', '## 下一步边界',
              '先根据本表决定需要修正的尺度/评价口径，不因本轮诊断自动上2万或补全套训练。',
              '边界敏感性分数变化不证明新库差距的因果；不能重定义测试真值后冒充原测试成绩提高。',
              '若固定个例显示新的内部恢复问题，正式量化应复用旧权重对原完整测试集推理，而不是重新训练；是否做该步取决于论文主张。']
    (OUT/'内部恢复与尺度诊断_CN.md').write_text('\n'.join(lines),encoding='utf8')
    for file in [__file__,'test_internal_recovery_audit.py']:
        source=track(file); (OUT/source.name).write_bytes(source.read_bytes())
    write(OUT/'manifest.json',dict(complete=True,inputs=inputs,checks=checks,sealed_test_used=False,
          outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and 'matplotlib_cache' not in p.parts}))
    print(json.dumps(dict(output=str(OUT),scale=scale_summary,theory=[r for r in theory_summary if r['region']=='core1']),ensure_ascii=False,indent=2))


def make_figures(figs,rows,oldrows,scale):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'pdf.fonttype':42,'font.size':10})
    colors=['#0072B2','#D55E00','#009E73']
    def save(fig,name):
        fig.savefig(figs/(name+'.png'),dpi=200); fig.savefig(figs/(name+'.pdf')); plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    names=[r['name'] for r in rows if r['arm']==ARM_NAMES[0] and r['law']=='oblique_gradient' and r['region']=='support']
    for ax,(law,region,key,title) in zip(axs.flat,[('oblique_gradient','support','centered_rmse','渐变真值：体内去均值误差（全部支撑）'),
            ('oblique_gradient','core1','centered_rmse','渐变真值：去一层边界后的误差'),
            ('constant','support','prediction_fluctuation_rms','常值真值：预测产生的虚假区内起伏'),
            ('oblique_gradient','support','fluctuation_skill','渐变真值：区内起伏技能（>0优于零起伏）')]):
        for a,name in enumerate(ARM_NAMES):
            data=[r for r in rows if r['arm']==name and r['law']==law and r['region']==region]
            y=[np.nan if r[key] is None else r[key] for r in data]
            ax.plot(range(6),y,'o-',label=name,color=colors[a])
        ax.set_title(title); ax.set_xticks(range(6)); ax.set_xticklabels(names,rotation=20,ha='right')
        ax.set_ylabel('无量纲' if key=='fluctuation_skill' else 'g/cm³'); ax.grid(alpha=.2); ax.legend(fontsize=8)
        if key=='fluctuation_skill': ax.axhline(0,color='gray',ls='--')
    fig.suptitle('已见理论诊断｜固定120轮；按真值体分别去均值，不作独立泛化结论',fontsize=14)
    save(fig,'01_内部起伏恢复')
    fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    x=np.arange(6); names=[s['name'] for s in scale]
    axs[0].bar(x,[s['median_pure_full_fraction'] for s in scale],color=colors[0])
    axs[0].set(title='固定60训练例：纯分区满格比例中位数',ylabel='占支撑格数的比例',ylim=(0,1))
    axs[1].bar(x,[s['median_noise_ratio_2pct'] for s in scale],color=colors[1])
    axs[1].axhline(1,color='gray',ls='--'); axs[1].set(title='去掉区内变化后的正演差值',ylabel='差值RMSE / 2%峰值噪声σ')
    for ax in axs: ax.set_xticks(x); ax.set_xticklabels(names,rotation=25,ha='right'); ax.grid(axis='y',alpha=.2)
    fig.suptitle('同占据率、同分区质量配对｜未训练常值网络；噪声比值不是可恢复性检验',fontsize=13)
    save(fig,'02_尺度与配对正演差')


if __name__ == '__main__': run()
