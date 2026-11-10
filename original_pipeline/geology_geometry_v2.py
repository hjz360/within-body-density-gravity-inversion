"""Bounded geometry acceptance pilot; not a production geological training set."""
import os
os.environ.setdefault('MPLBACKEND','Agg')
import json,sys,hashlib
from pathlib import Path
from functools import lru_cache
import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation
from geology_gallery import FAMILIES,NAMES,LAWS,make_model,X,Y,Z

@lru_cache(None)
def templates(family):
    labels=make_model(family,0,0)[1]
    return [ndimage.distance_transform_edt(labels==k)*50-ndimage.distance_transform_edt(labels!=k)*50 for k in range(1,int(labels.max())+1)]

def geometry_check(labels,expected,min_body_voxels=25,min_gap_m=50):
    reasons=[]; counts=[]; erosion=[]; gaps=[]
    full=labels>0
    for k in range(1,expected+1):
        mask=labels==k;n=int(mask.sum());counts.append(n)
        if n<min_body_voxels:reasons.append('body_too_small')
        if ndimage.label(mask)[1]!=1:reasons.append('body_not_face_connected')
        eroded=ndimage.binary_erosion(mask)
        erosion.append(int(eroded.sum()))
        if eroded.sum()<3:reasons.append('no_resolved_interior')
        for j in range(1,k):
            # Conservative lower bound on gap between voxel cubes.
            d=ndimage.distance_transform_edt(labels!=j,sampling=50)
            gap=float(d[mask].min()-50*np.sqrt(3)) if n else -1
            gaps.append(gap)
            if gap<min_gap_m:reasons.append('gap_lower_bound_below_configured_min')
    if ndimage.label(full,np.ones((3,3,3)))[1]!=expected:reasons.append('wrong_component_count')
    touch=bool(full[0].any() or full[-1].any() or full[:,0].any() or full[:,-1].any() or full[:,:,0].any() or full[:,:,-1].any())
    return {'passed':not reasons,'reasons':sorted(set(reasons)),'body_voxels':counts,'eroded_body_voxels':erosion,'gap_lower_bound_m':gaps,'boundary_contact':touch}

def generate(family,index):
    seed=20260921+FAMILIES.index(family)*10000+index
    rng=np.random.default_rng(seed); signed=templates(family);rejected=[]
    for attempt in range(150):
        angles=[float(rng.uniform(-12,12)),float(rng.uniform(-12,12)),float(rng.uniform(-180,180))]
        scale=rng.uniform(.75,1.2,3);shift=rng.uniform([-360,-360,-100],[360,360,100])
        xyz=np.stack([X-800-shift[0],Y-800-shift[1],Z-400-shift[2]],axis=0).reshape(3,-1)
        local=(Rotation.from_euler('xyz',angles,degrees=True).as_matrix().T@xyz)/scale[:,None]
        local=local.reshape((3,)+X.shape)+np.array([800,800,400])[:,None,None,None]
        q=(local[[2,1,0]]/50)-.5
        perturb=ndimage.gaussian_filter(rng.normal(size=X.shape),sigma=2.0)
        perturb=perturb/max(perturb.std(),1e-12)
        amp=float(rng.uniform(8,28));labels=np.zeros(X.shape,np.uint8);overlap=False
        for k,s in enumerate(signed,1):
            mask=ndimage.map_coordinates(s,q,order=1,mode='constant',cval=-1000)+amp*perturb>0
            overlap|=bool((labels[mask]>0).any());labels[mask]=k
        audit=geometry_check(labels,len(signed))
        if overlap:audit['passed']=False;audit['reasons'].append('overlap')
        if audit['passed']:
            audit.update(family=family,index=index,seed=seed,attempts=attempt+1,rejected=rejected,angles_deg=angles,scale=scale.tolist(),translation_m=shift.tolist(),boundary_perturbation_rms_m=amp)
            return labels,local,audit
        rejected.append(audit['reasons'])
    raise RuntimeError(f'No accepted sample: {family}/{index}; {rejected[-3:]}')

def assign_density(labels,coords,index,law,mean_range=(.45,.85),amplitude=.24):
    rng=np.random.default_rng(90000+index);m=np.zeros(labels.shape,np.float32)
    for k in range(1,int(labels.max())+1):
        mask=labels==k;mean=rng.uniform(*mean_range)
        xyz=[(p-p[mask].mean())/max(np.ptp(p[mask]),50) for p in coords];x,y,z=xyz
        if law==0:q=np.zeros_like(X)
        elif law==1:q=z
        elif law==2:q=np.exp(-((x-.2)**2+y*y+z*z)/.08)
        elif law==3:q=np.exp(-((x-.25)**2+y*y+z*z)/.045)+np.exp(-((x+.25)**2+y*y+z*z)/.045)
        else:
            noise=np.random.default_rng(123000+index*7+k).normal(size=X.shape)
            q=ndimage.gaussian_filter(noise,(1.2,2,2))
        q=q-q[mask].mean();q=q/max(abs(q[mask]).max(),1e-12)
        m[mask]=mean+amplitude*q[mask]
    return m

def run(out):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from collections import Counter
    sys.path.insert(0,r'F:\2026.3\代码')
    from gravity_validation import load_hdf5_kernel
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    kernel=load_hdf5_kernel(r'F:\2026.3\代码\G.mat')*1000
    density=[];bodies=[];records=[]
    for fi,f in enumerate(FAMILIES):
        for i in range(20):
            l,c,a=generate(f,i);m=assign_density(l,c,fi*20+i,i%5)
            assert np.array_equal(m>0,l>0) and np.isfinite(m).all() and m.min()>=0 and m.max()<=1.2
            a['density_law']=LAWS[i%5];a['mass_kg']=float(m.astype(float).sum()*1000*50**3)
            a['centroid_xyz_m']=[float(v[l>0].mean()) for v in [X,Y,Z]]
            density.append(m);bodies.append(l);records.append(a)
    density=np.array(density);gravity=(kernel@density.reshape(120,-1).T).T.reshape(120,32,32)
    assert np.isfinite(gravity).all() and gravity.min()>0
    np.savez_compressed(out/'geometry_acceptance_120.npz',density_zyx=density,body_labels_zyx=np.array(bodies),gravity_yx_mgal=gravity)
    (out/'metadata.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    rejected=Counter(reason for r in records for reasons in r['rejected'] for reason in reasons)
    report={'accepted':120,'candidate_count':sum(r['attempts'] for r in records),'rejected_reason_counts':dict(rejected),'boundary_contact_count':sum(r['boundary_contact'] for r in records),'minimum_body_voxels':min(min(r['body_voxels']) for r in records),'minimum_gap_lower_bound_m':min(g for r in records for g in r['gap_lower_bound_m']),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'status':'geometry acceptance only; no training; no field calibration','limitations':['Warped discrete prototype, not geological-history simulation','Boundary-touching samples not yet audited for retained original body mass','Erosion core check does not prove every connecting neck is sufficiently thick','120 samples do not establish distribution coverage','Local gradient coordinates; random density noise remains grid-oriented','Cell-center representation, no subcell integration']}
    (out/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei'],'axes.unicode_minus':False,'font.size':10})
    norm=Normalize(0,1.2);cmap=plt.get_cmap('viridis');edges=np.meshgrid(np.arange(33)*50,np.arange(33)*50,np.arange(17)*50,indexing='ij')
    for fi,f in enumerate(FAMILIES):
        fig=plt.figure(figsize=(15,10),layout='constrained');axs=[]
        fig.suptitle(NAMES[fi]+'：随机几何验收样本（非反演结果）',fontsize=18)
        for col,i in enumerate([0,6,13,19]):
            idx=fi*20+i;m=density[idx];a=m.transpose(2,1,0)
            ax=fig.add_subplot(3,4,col+1,projection='3d');axs.append(ax)
            ax.voxels(*edges,a>0,facecolors=cmap(norm(a)),edgecolors=None,alpha=.85)
            ax.set(xlim=(0,1600),ylim=(0,1600),zlim=(800,0),xlabel='东向 / m',ylabel='北向 / m',zlabel='深度 / m',title=f'样本 {i}｜{LAWS[i%5]}')
            ax.view_init(24,-58);ax.set_box_aspect((2,2,1));ax.set_xticks([0,800,1600]);ax.set_yticks([0,800,1600]);ax.set_zticks([0,400,800])
            ax=fig.add_subplot(3,4,col+5);axs.append(ax)
            iy=int(np.argmax((m>0).sum(axis=(0,2))))
            ax.imshow(m[:,iy,:],extent=(0,1600,800,0),norm=norm,cmap=cmap,interpolation='nearest');ax.set(title=f'北向 {(iy+.5)*50:.0f} m 剖面',xlabel='东向 / m',ylabel='深度 / m')
            ax=fig.add_subplot(3,4,col+9);im=ax.imshow(gravity[idx],origin='lower',extent=(0,1600,0,1600),cmap=cmap);ax.set(title='无噪声正演异常',xlabel='东向 / m',ylabel='北向 / m');fig.colorbar(im,ax=ax,shrink=.65,label='mGal')
        fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=axs,shrink=.55,label='密度差 / (g/cm³)')
        for ext in ['png','pdf']:fig.savefig(out/f'图{fi+1:02d}_{NAMES[fi]}.{ext}',dpi=170,bbox_inches='tight')
        plt.close(fig)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':run(sys.argv[1])
