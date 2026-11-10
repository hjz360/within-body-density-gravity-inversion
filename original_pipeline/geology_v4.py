"""Coverage revision: preserve v3 six families and add four connected families.

No observed field data or diagnostic test model arrays are used here.
"""
import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation
from geology_gallery import FAMILIES as OLD_FAMILIES, NAMES as OLD_NAMES
from geology_v3 import generate as old_generate, EX, EY, EZ, CROP, boundary_flags
from geology_geometry_v2 import geometry_check, assign_density

NEW_FAMILIES=['compact_single','continuous_slab','curved_sheet','connected_branches']
FAMILIES=OLD_FAMILIES+NEW_FAMILIES
NAMES=OLD_NAMES+['不规则紧凑单体','连续倾斜板','弯曲连续层','连通多叶体']
EXPECTED_COMPONENTS=dict(zip(FAMILIES,[2,3,2,1,1,2,1,1,1,1]))

def make_plan(ntrain=10000,nval=1200,ntest=400):
    rows=[]
    for si,(split,n) in enumerate([('train',ntrain),('validation',nval),('test_sealed',ntest)]):
        assert n%50==0
        for i in range(n):
            f=FAMILIES[i%10];idx=1000000+si*100000+i//10
            rows.append(dict(planned_row=len(rows),family=f,geometry_index=idx,
                group_key=f'{f}:{idx}',density_seed_index=4000000+si*100000+i,
                density_law_index=(i//10)%5,split=split))
    return rows

def generate(family,index):
    if family in OLD_FAMILIES:
        l,c,a=old_generate(family,index)
        a['generator_branch']='v3_preserved'
        return l,c,a
    if family not in NEW_FAMILIES:raise ValueError(family)
    rng=np.random.default_rng(70000000+NEW_FAMILIES.index(family)*10000000+index)
    rejected=[]
    for attempt in range(200):
        angles=rng.uniform([-12,-12,-180],[12,12,180])
        xyz=np.stack([EX-800,EY-800,EZ-400]).reshape(3,-1)
        local=(Rotation.from_euler('xyz',angles,degrees=True).as_matrix().T@xyz).reshape((3,)+EX.shape)
        u,v,w=local
        axes=rng.uniform([230,170,90],[390,290,180]);ax,ay,az=axes
        power=float(rng.uniform(1.6,3.8));slope=float(rng.uniform(-.7,.7));bend=float(rng.uniform(40,100))
        if family=='compact_single':
            field=min(axes)*(1-(abs(u/ax)**power+abs(v/ay)**power+abs(w/az)**power)**(1/power))
        elif family in ('continuous_slab','curved_sheet'):
            ax=float(rng.uniform(300,460));ay=float(rng.uniform(190,320));az=float(rng.uniform(85,125))
            center=slope*u
            if family=='curved_sheet':center=center+bend*((u/ax)**2-.35)+30*np.sin(v/ay*2)
            field=np.minimum(min(ax,ay)*(1-(abs(u/ax)**power+abs(v/ay)**power)**(1/power)),az-abs(w-center))
        else:
            # Overlapping, tapered lobes form one connected branching body.
            def ell(cx,cy,cz,rx,ry,rz):
                return min(rx,ry,rz)*(1-np.sqrt(((u-cx)/rx)**2+((v-cy)/ry)**2+((w-cz)/rz)**2))
            separation=float(rng.uniform(110,190))
            field=np.maximum.reduce([ell(-80,0,0,300,140,az),ell(140,separation,30,230,170,az),ell(140,-separation,-30,230,170,az)])
        noise=ndimage.gaussian_filter(rng.normal(size=EX.shape),sigma=2)
        amp=float(rng.uniform(5,18));field+=amp*noise/max(noise.std(),1e-12)
        full=(field>0).astype(np.uint8)
        points=np.argwhere(full);lo=points.min(0);hi=points.max(0);span=hi-lo+1
        # Select position over every feasible voxel translation, not central window.
        margins=np.array([1,0,0]);shape=np.array([16,32,32])
        if np.any(span>shape-2*margins):rejected.append(['extent_exceeds_grid']);continue
        target=np.array([rng.integers(margins[k],shape[k]-margins[k]-span[k]+1) for k in range(3)])+16
        shift=target-lo
        shifted=np.roll(full,tuple(shift),axis=(0,1,2))
        flags=boundary_flags(shifted)
        if any(flags.values()):rejected.append(['outside_grid']);continue
        l=shifted[CROP].copy();a=geometry_check(l,1)
        if not a['passed']:rejected.append(a['reasons']);continue
        coords=np.roll(local,tuple(shift),axis=(1,2,3))[(slice(None),)+CROP].copy()
        assert l.sum()==full.sum()
        a.update(family=family,index=index,attempts=attempt+1,rejected=rejected,
            generator_branch='v4_parametric_connected',angles_deg=angles.tolist(),
            shape_parameters=dict(axes_xyz_m=[float(ax),float(ay),float(az)],power=power,slope=slope,bend_m=bend),
            translation_zyx_m=(shift*50).tolist(),boundary_perturbation_rms_m=amp,
            boundary_flags=flags,body_volume_retention=[1.0])
        return l,coords,a
    raise RuntimeError(f'{family}/{index} failed: {rejected[-3:]}')

def model(row):
    l,c,a=generate(row['family'],row['geometry_index'])
    m=assign_density(l,c,row['density_seed_index'],row['density_law_index'])
    return m,l,a
