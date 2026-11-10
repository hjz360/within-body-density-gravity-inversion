"""Small, non-training geological-shape prototype. Arrays use Z,Y,X order."""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys, json, hashlib
from pathlib import Path
import numpy as np
from scipy import ndimage

FAMILIES = ['separated_pair','separated_triple','stacked','lens','lens_keel','faulted_layers']
NAMES = ['分离双体','分离三体','上下叠置体','变厚度透镜体','主体与倾斜延伸体','断错分段层状体']
LAWS = ['近均匀','垂向渐变','偏心高密度区','双高密度区','相关密度起伏']
Z,Y,X = np.meshgrid((np.arange(16)+.5)*50,(np.arange(32)+.5)*50,(np.arange(32)+.5)*50,indexing='ij')

def make_model(family, variant, law=None):
    rng = np.random.default_rng(20260920 + FAMILIES.index(family)*100 + variant)
    dx,dy = rng.uniform(-65,65,2)
    x,y,z = X-dx,Y-dy,Z
    def ell(cx,cy,cz,ax,ay,az):
        return ((x-cx)/ax)**2+((y-cy)/ay)**2+((z-cz)/az)**2 <= 1
    if family=='separated_pair':
        masks=[ell(430,700,260,190,260,130),ell(1130,850,420,240,210,160)]
    elif family=='separated_triple':
        masks=[ell(350,450,230,180,180,120),ell(1150,450,380,210,180,150),ell(800,1130,510,240,210,160)]
    elif family=='stacked':
        masks=[ell(730,770,200,270,230,100),ell(900,850,560,250,220,110)]
    elif family=='lens':
        u=x-800; v=y-800
        t=110*(1-.3*np.tanh(u/300))
        center=330+.18*u+35*np.sin(v/200)
        masks=[(u/530)**2+(v/290)**2+((z-center)/t)**2 <= 1]
    elif family=='lens_keel':
        main=ell(900,830,260,400,290,135)
        keel=((x-(890-.65*(z-280)))/110)**2+((y-810)/130)**2 <= 1
        masks=[main | (keel & (z>=260) & (z<=660))]
    else:
        center=280+.12*(x-800)+np.where(x>800,220,0)
        footprint=((x-800)/540)**2+((y-800)/340)**2<=1
        masks=[footprint & (x<710) & (abs(z-center)<65),footprint & (x>890) & (abs(z-center)<65)]
    labels=np.zeros(X.shape,np.uint8)
    m=np.zeros(X.shape,np.float32)
    law=variant if law is None else law
    props=[]
    for k,mask in enumerate(masks,1):
        assert not np.any(labels[mask]), 'Bodies overlap'
        labels[mask]=k
        # Each body receives an independent mean, independent of geometry RNG.
        prng=np.random.default_rng(82000+FAMILIES.index(family)*100+variant*5+k)
        mean=float(prng.uniform(.45,.85))
        xn=(X-X[mask].mean())/max(np.ptp(X[mask]),50)
        zn=(Z-Z[mask].mean())/max(np.ptp(Z[mask]),50)
        yn=(Y-Y[mask].mean())/max(np.ptp(Y[mask]),50)
        if law==0: q=np.zeros_like(X)
        elif law==1: q=zn
        elif law==2: q=np.exp(-((xn-.22)**2+yn**2+zn**2)/.08)
        elif law==3:
            q=np.exp(-((xn-.25)**2+yn**2+zn**2)/.045)+np.exp(-((xn+.25)**2+yn**2+zn**2)/.045)
        else: q=ndimage.gaussian_filter(prng.normal(size=X.shape),sigma=(1.2,2,2))
        q=q-q[mask].mean()
        q=q/max(np.max(abs(q[mask])),1e-12)
        m[mask]=mean+.24*q[mask]
        props.append({'body_id':k,'mean_density_g_cm3':mean,'voxels':int(mask.sum())})
    meta={'family':family,'variant':variant,'density_law':LAWS[law], 'body_properties':props,
          'components_26_positive':int(ndimage.label(m>0,np.ones((3,3,3)))[1]),
          'mass_kg':float(m.astype(float).sum()*1000*50**3),
          'geometry_translation_m':[float(dx),float(dy)],'status':'illustrative prototype, not training or field truth'}
    return m,labels,meta

def main(out):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    sys.path.insert(0,r'F:\2026.3\代码')
    from gravity_validation import load_hdf5_kernel, rectangular_prism_gz, observation_grid
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10})
    kernel=load_hdf5_kernel(Path(r'F:\2026.3\代码\G.mat'))*1000
    # Independent analytical prism spot-check: off-centre + deep voxels.
    xo,yo,zo=observation_grid()
    errors=[]
    for iz,iy,ix in [(2,7,23),(11,21,8),(5,15,15)]:
        index=iz*1024+iy*32+ix
        independent=rectangular_prism_gz(xo,yo,zo,(ix*50,(ix+1)*50,iy*50,(iy+1)*50,iz*50,(iz+1)*50),1000)
        error=float(np.max(abs(independent-kernel[:,index])))
        errors.append(error)
        assert np.allclose(independent,kernel[:,index],rtol=1e-5,atol=1e-7),error
    models=[]; labels=[]; meta=[]; gravity=[]
    for f in FAMILIES:
        for v in range(5):
            m,l,r=make_model(f,v); g=(kernel@m.ravel()).reshape(32,32)
            assert np.isfinite(g).all() and g.min()>0
            models.append(m);labels.append(l);meta.append(r);gravity.append(g)
    np.savez_compressed(out/'prototype_models.npz',density_zyx=np.array(models),body_labels_zyx=np.array(labels),gravity_yx_mgal=np.array(gravity))
    norm=Normalize(0,1.2);cmap=plt.get_cmap('viridis')
    def volume(ax,m):
        a=m.transpose(2,1,0)
        ax.voxels(*np.meshgrid(np.arange(33)*50,np.arange(33)*50,np.arange(17)*50,indexing='ij'),a>0,facecolors=cmap(norm(a)),edgecolors=None,alpha=.82)
        ax.set(xlim=(0,1600),ylim=(0,1600),zlim=(800,0),xlabel='东向 / m',ylabel='北向 / m',zlabel='深度 / m')
        ax.set_box_aspect((2,2,1));ax.view_init(24,-58)
        ax.set_xticks([0,800,1600]);ax.set_yticks([0,800,1600]);ax.set_zticks([0,400,800]);ax.tick_params(labelsize=8)
    def save(fig,name):
        fig.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight')
        fig.savefig(out/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    for fi,f in enumerate(FAMILIES):
        fig=plt.figure(figsize=(19,11),layout='constrained')
        density_axes=[]
        fig.suptitle(NAMES[fi]+'：几何原型与内部密度变化（非反演结果）',fontsize=19)
        for v in range(5):
            i=fi*5+v;m=models[i]
            ax=fig.add_subplot(3,5,v+1,projection='3d');volume(ax,m);ax.set_title(LAWS[v])
            density_axes.append(ax)
            ax=fig.add_subplot(3,5,v+6)
            density_axes.append(ax)
            iy=int(np.argmax(m.sum(axis=(0,2))))
            ax.imshow(m[:,iy,:],extent=(0,1600,800,0),cmap=cmap,norm=norm,aspect='equal',interpolation='nearest')
            ax.set(title=f'北向 {(iy+.5)*50:.0f} m 剖面',xlabel='东向 / m',ylabel='深度 / m')
            ax=fig.add_subplot(3,5,v+11)
            im=ax.imshow(gravity[i],origin='lower',extent=(0,1600,0,1600),cmap='viridis',interpolation='nearest')
            ax.set(title='无噪声正演异常',xlabel='东向 / m',ylabel='北向 / m');fig.colorbar(im,ax=ax,shrink=.75,label='mGal')
        fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=density_axes,shrink=.55,label='密度差 / (g/cm³)')
        save(fig,f'图{fi+1:02d}_{NAMES[fi]}')
    fig=plt.figure(figsize=(16,10),layout='constrained')
    fig.suptitle('六类模型构建原型｜相同视角、相同密度色标',fontsize=20)
    for i in range(6):
        ax=fig.add_subplot(2,3,i+1,projection='3d');volume(ax,models[i*5+2]);ax.set_title(NAMES[i])
    fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=fig.axes,shrink=.65,label='密度差 / (g/cm³)')
    save(fig,'图00_六类三维模型总览')
    # One fixed geometry and fixed body means; density pattern alone changes.
    fig=plt.figure(figsize=(18,7),layout='constrained');variants=[]
    fig.suptitle('固定分离双体几何：仅改变内部密度分布',fontsize=19)
    for law in range(5):
        m,l,r=make_model('separated_pair',0,law=law);variants.append(m)
        ax=fig.add_subplot(2,5,law+1,projection='3d');volume(ax,m);ax.set_title(LAWS[law])
        ax=fig.add_subplot(2,5,law+6);iy=15
        ax.imshow(m[:,iy,:],extent=(0,1600,800,0),norm=norm,cmap=cmap,interpolation='nearest');ax.set(xlabel='东向 / m',ylabel='深度 / m',title='北向 775 m 剖面')
    fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=fig.axes,shrink=.6,label='密度差 / (g/cm³)')
    save(fig,'图07_同几何不同密度规律')
    np.save(out/'fixed_geometry_density_laws.npy',np.array(variants))
    audit={'prototype_count':30,'components_by_family':{f:[r['components_26_positive'] for r in meta if r['family']==f] for f in FAMILIES},
      'independent_prism_spotcheck_max_abs_error_mgal':errors,'array_order':'ZYX; gravity YX',
      'grid_m':[50,50,50],'density_unit':'g/cm3 contrast','observation_height_z_m':-.1,
      'limitations':['Only 30 handcrafted geometry variants; no coverage claim','No field outlines fitted or used as truth','Positive contrasts only; no host-density structure yet','Cell-center discretization, not subvoxel integration','No training or inversion run','Within each family geometry and density vary together; figure07 separately fixes geometry'],
      'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'kernel_sha256':hashlib.sha256(Path(r'F:\2026.3\代码\G.mat').read_bytes()).hexdigest()}
    (out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main(sys.argv[1])
