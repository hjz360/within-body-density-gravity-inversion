"""Full-domain-first, fully buried geometry pilot. Not a production dataset."""
import os
os.environ.setdefault('MPLBACKEND','Agg')
import sys,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation
from geology_geometry_v2 import templates,geometry_check,assign_density
from geology_gallery import FAMILIES,NAMES,X,Y,Z

PAD=16
CROP=(slice(16,32),slice(16,48),slice(16,48))
EZ,EY,EX=np.meshgrid((np.arange(-16,32)+.5)*50,(np.arange(-16,48)+.5)*50,(np.arange(-16,48)+.5)*50,indexing='ij')

def boundary_flags(labels):
    m=labels>0
    return {'above_surface':bool(m[:16].any()),'below_bottom':bool(m[32:].any()),
        'outside_horizontal':bool(m[:,:16].any() or m[:,48:].any() or m[:,:,:16].any() or m[:,:,48:].any()),
        'outer_grid_contact':bool(m[0].any() or m[-1].any() or m[:,0].any() or m[:,-1].any() or m[:,:,0].any() or m[:,:,-1].any())}

def generate(family,index,config=None):
    if config is None:
        config={'rotation_xyz_deg_ranges':[[-12,12],[-12,12],[-180,180]],'scale_range':[.75,1.2],
          'translation_xyz_m_ranges':[[-360,360],[-360,360],[-100,100]],'boundary_noise_sigma_voxels':2,
          'boundary_noise_rms_m_range':[8,28],'max_attempts':200,'min_body_voxels':25,'min_pair_gap_lower_bound_m':50}
    seed=20260922+FAMILIES.index(family)*10000+index
    rng=np.random.default_rng(seed);signed=templates(family);reject=[]
    for attempt in range(config['max_attempts']):
        angles=[float(rng.uniform(*bounds)) for bounds in config['rotation_xyz_deg_ranges']]
        scale=rng.uniform(*config['scale_range'],size=3)
        bounds=np.array(config['translation_xyz_m_ranges']);shift=rng.uniform(bounds[:,0],bounds[:,1])
        xyz=np.stack([EX-800-shift[0],EY-800-shift[1],EZ-400-shift[2]]).reshape(3,-1)
        local=(Rotation.from_euler('xyz',angles,degrees=True).as_matrix().T@xyz)/scale[:,None]
        local=local.reshape((3,)+EX.shape)+np.array([800,800,400])[:,None,None,None]
        q=local[[2,1,0]]/50-.5
        noise=ndimage.gaussian_filter(rng.normal(size=EX.shape),sigma=config['boundary_noise_sigma_voxels'])
        noise/=max(noise.std(),1e-12);amp=float(rng.uniform(*config['boundary_noise_rms_m_range']))
        full=np.zeros(EX.shape,np.uint8);overlap=False
        for k,s in enumerate(signed,1):
            mask=ndimage.map_coordinates(s,q,order=1,mode='constant',cval=-1000)+amp*noise>0
            overlap|=bool((full[mask]>0).any());full[mask]=k
        flags=boundary_flags(full);reasons=[k for k,v in flags.items() if v]
        if overlap:reasons.append('overlap')
        if reasons:reject.append(reasons);continue
        cropped=full[CROP].copy();a=geometry_check(cropped,len(signed),config['min_body_voxels'],config['min_pair_gap_lower_bound_m'])
        if not a['passed']:reject.append(a['reasons']);continue
        retention=[float((cropped==k).sum()/(full==k).sum()) for k in range(1,len(signed)+1)]
        assert all(v==1.0 for v in retention)
        a.update(family=family,index=index,seed=seed,attempts=attempt+1,rejected=reject,boundary_flags=flags,
          body_volume_retention=retention,angles_deg=angles,scale=scale.tolist(),translation_m=shift.tolist(),boundary_perturbation_rms_m=amp,
          full_label_sha256=hashlib.sha256(full.tobytes()).hexdigest())
        return cropped,local[(slice(None),)+CROP],a
    raise RuntimeError(f'{family}/{index}: maximum attempts exceeded')

def run(out):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    sys.path.insert(0,r'F:\2026.3\代码')
    from gravity_validation import load_hdf5_kernel
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    density=[];labels=[];records=[]
    for fi,f in enumerate(FAMILIES):
        for i in range(5):
            l,c,r=generate(f,i);m=assign_density(l,c,fi*5+i,i)
            assert np.array_equal(m>0,l>0) and np.isfinite(m).all()
            r['centroid_xyz_m']=[float(p[l>0].mean()) for p in [X,Y,Z]]
            r['body_mass_retention']=[1.0]*int(l.max())
            r['mass_retention_reason']='No active voxel outside crop; no density renormalization between full and crop needed.'
            density.append(m);labels.append(l);records.append(r)
    density=np.array(density);kernel=load_hdf5_kernel(r'F:\2026.3\代码\G.mat')*1000
    gravity=(kernel@density.reshape(30,-1).T).T.reshape(30,32,32)
    assert np.isfinite(gravity).all() and gravity.min()>0
    np.savez_compressed(out/'complete_bodies_30.npz',density_zyx=density,body_labels_zyx=np.array(labels),gravity_yx_mgal=gravity)
    rejected=Counter(reason for r in records for reasons in r['rejected'] for reason in reasons)
    centroid=np.array([r['centroid_xyz_m'] for r in records])
    summary={'accepted':30,'candidate_count':sum(r['attempts'] for r in records),'rejected_reason_counts':dict(rejected),
      'all_body_volume_retention_1':all(all(v==1 for v in r['body_volume_retention']) for r in records),
      'all_boundary_flags_false':all(not any(r['boundary_flags'].values()) for r in records),
      'centroid_xyz_min_m':centroid.min(0).tolist(),'centroid_xyz_max_m':centroid.max(0).tolist(),
      'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'limitations':['50m discrete geometry, not subvoxel truth','Same six prototype shapes, not geological coverage','Full-body selection changes effective position distribution','Thin-structure screen not resolved by this change','No field petrophysical calibration','No training started']}
    (out/'metadata.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei'],'axes.unicode_minus':False,'font.size':10})
    fig=plt.figure(figsize=(16,11),layout='constrained');axes=[];norm=Normalize(0,1.2);cmap=plt.get_cmap('viridis')
    edges=np.meshgrid(np.arange(33)*50,np.arange(33)*50,np.arange(17)*50,indexing='ij')
    for fi,f in enumerate(FAMILIES):
        i=fi*5+2;a=density[i].transpose(2,1,0)
        row,col=divmod(fi,3)
        ax=fig.add_subplot(4,3,row*6+col+1,projection='3d');axes.append(ax)
        ax.voxels(*edges,a>0,facecolors=cmap(norm(a)),alpha=.85,edgecolors=None)
        ax.set(xlim=(0,1600),ylim=(0,1600),zlim=(800,0),title=NAMES[fi],xlabel='东向 / m',ylabel='北向 / m',zlabel='深度 / m')
        ax.set_box_aspect((2,2,1));ax.view_init(24,-58);ax.set_xticks([0,800,1600]);ax.set_yticks([0,800,1600]);ax.set_zticks([0,400,800])
        ax=fig.add_subplot(4,3,row*6+col+4);im=ax.imshow(gravity[i],origin='lower',extent=(0,1600,0,1600),cmap='viridis');ax.set(xlabel='东向 / m',ylabel='北向 / m',title='对应无噪声重力异常');fig.colorbar(im,ax=ax,shrink=.7,label='mGal')
    fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=axes,shrink=.7,label='密度差 / (g/cm³)')
    fig.suptitle('扩展域先生成、完整体再裁剪｜六类随机样本（非反演结果）',fontsize=18)
    for ext in ['png','pdf']:fig.savefig(out/f'六类完整体与重力异常.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':run(sys.argv[1])
