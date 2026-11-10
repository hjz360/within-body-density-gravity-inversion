"""VMS engineering prototype v2; independent parameter streams, analytic z integral.
Ranges are engineering assumptions, not calibrated geological distributions.
Density is continuous within zones; geological interfaces may be discontinuous.
"""
import hashlib,json
import numpy as np
SPLITS={'train':0,'validation':1,'test_reserved':2}
def rng(split,f,i,stream):
    return np.random.default_rng(np.random.SeedSequence([20260924,2,SPLITS[split],f,i,stream]))
def sample(split,f,i):
    if split not in SPLITS or f not in [0,1,2] or not isinstance(i,int) or i<0:raise ValueError('invalid sample key')
    r=rng(split,f,i,0)
    g=dict(family=f,slope=float(r.uniform(-.30,.30)))
    if f==0:g.update(ax=float(r.uniform(360,480)),ay=float(r.uniform(190,270)),half=float(r.uniform(15,40)),stock_ax=float(r.uniform(170,250)),stock_ay=float(r.uniform(120,190)),stock_offset=float(r.uniform(-80,80)),stock_side=int(r.choice([-1,1])),stock_extent=float(r.uniform(80,120)))
    elif f==1:g.update(ax=float(r.uniform(170,230)),ay=float(r.uniform(180,260)),half=float(r.uniform(65,125)),separation_half=float(r.uniform(310,390)),depth_separation=float(r.uniform(140,220)),ratio_x=float(r.uniform(.7,1.15)),ratio_y=float(r.uniform(.7,1.15)),ratio_z=float(r.uniform(.7,1.15)),slope2=float(r.uniform(-.3,.3)))
    else:g.update(ax=float(r.uniform(430,560)),ay=float(r.uniform(230,320)),half=float(r.uniform(15,40)),bend=float(r.uniform(25,60)),throw=float(r.uniform(100,200)),gap=float(r.uniform(70,110)),bend_scale=float(r.uniform(180,300)),bend_phase=float(r.uniform(-np.pi,np.pi)))
    r=rng(split,f,i,1);ds=[];base=float(r.uniform(.7,1.3))
    for zone in range(2):
        ds.append(dict(base=base*(float(r.uniform(.2,.5)) if f==0 and zone==1 else (float(r.uniform(.75,1.25)) if zone else 1)),
            amplitude=float(r.uniform(.05,.22)),vertical=float(r.uniform(0,.12)),trend=float(r.uniform(-.25,.25)),
            lu=float(r.uniform(150,350)),lv=float(r.uniform(150,350)),lz=float(r.uniform(100,300)),
            phase_u=float(r.uniform(-np.pi,np.pi)),phase_v=float(r.uniform(-np.pi,np.pi)),phase_z=float(r.uniform(-np.pi,np.pi))))
    return dict(group_id=f'v2-{split}-{f}-{i:05d}',seed_key=f'20260924:2:{SPLITS[split]}:{f}:{i}',split=split,family=f,index=i,geometry=g,density=ds,
        geometry_sha256=hashlib.sha256(json.dumps(g,sort_keys=True).encode()).hexdigest())
def bounds(g):
    s=abs(g['slope']);a=g['ax'];b=g['ay'];h=g['half'];f=g['family']
    if f==0:
        a=max(a,abs(g['stock_offset'])+g['stock_ax']);b=max(b,g['stock_ay']);e=np.sqrt(1+s*s)*(h+g['stock_extent'])
        return a,b,310-s*a-e,310+s*a+e
    if f==1:
        aa=max(a,a*g['ratio_x']);bb=max(b,b*g['ratio_y']);hh=max(h,h*g['ratio_z']);ss=max(s,abs(g['slope2']))
        return g['separation_half']+aa,bb,250-ss*aa-hh,250+g['depth_separation']+ss*aa+hh
    return a,b,280-s*a-g['bend']-h,280+s*a+g['bend']+g['throw']+h
def pose(row):
    r=rng(row['split'],row['family'],row['index'],2);theta=float(r.uniform(-np.pi,np.pi));a,b,lo,hi=bounds(row['geometry'])
    rx=abs(np.cos(theta))*a+abs(np.sin(theta))*b;ry=abs(np.sin(theta))*a+abs(np.cos(theta))*b
    if max(rx,ry)>=775 or hi-lo>=750:raise ValueError('conservative bounds do not fit')
    dx=float(r.uniform(-775+rx,775-rx));dy=float(r.uniform(-775+ry,775-ry));dz=float(r.uniform(25-lo,775-hi))
    return dict(angle_rad=theta,translation=[dx,dy,dz],world_bounds=[[800+dx-rx,800+dy-ry,lo+dz],[800+dx+rx,800+dy+ry,hi+dz]])
def intervals(g,u,v):
    a,b,h=g['ax'],g['ay'],g['half'];s=g['slope'];f=g['family']
    if f==0:
        c=310+s*u;scale=np.sqrt(1+s*s);w=h*scale*np.sqrt(np.maximum(0,1-(u/a)**2-(v/b)**2));side=g['stock_side']
        p=c+side*.4*h*scale;q=c+side*(h+g['stock_extent'])*scale;lo=np.minimum(p,q);hi=np.maximum(p,q)
        valid=((u-g['stock_offset'])/g['stock_ax'])**2+(v/g['stock_ay'])**2<=1
        return c-w,c+w,lo,np.where(valid,hi,lo)
    if f==1:
        e=[]
        for j in range(2):
            x=(-1 if j==0 else 1)*g['separation_half'];aa=a*(g['ratio_x'] if j else 1);bb=b*(g['ratio_y'] if j else 1);hh=h*(g['ratio_z'] if j else 1)
            c=250+j*g['depth_separation']+(g['slope2'] if j else s)*(u-x);w=hh*np.sqrt(np.maximum(0,1-((u-x)/aa)**2-(v/bb)**2));e.extend([c-w,c+w])
        return e
    c=280+s*u+g['bend']*np.sin(v/g['bend_scale']+g['bend_phase'])+np.where(u>0,g['throw'],0);lo=c-h;valid=(u/a)**2+(v/b)**2<=1
    return lo,np.where(valid&(u < -g['gap']/2),c+h,lo),lo,np.where(valid&(u > g['gap']/2),c+h,lo)
def integral(lo,hi,u,v,d):
    """Exact integral of density over world z in [lo,hi], empty intervals zero."""
    hi=np.maximum(lo,hi);length=hi-lo
    horizontal=1+d['amplitude']*np.sin(u/d['lu']+d['phase_u'])*np.cos(v/d['lv']+d['phase_v'])
    return d['base']*(horizontal*length+d['trend']*((hi*hi-lo*lo)/1600-.5*length)+d['vertical']*d['lz']*(np.cos(lo/d['lz']+d['phase_z'])-np.cos(hi/d['lz']+d['phase_z'])))
def build(row,p,n=16):
    if not isinstance(n,int) or n<1:raise ValueError('invalid subdivision')
    g=row['geometry'];theta=p['angle_rad'];dx,dy,dz=p['translation'];q=(np.arange(n)+.5)*50/n;yy,xx=np.meshgrid(q,q,indexing='ij');xx=xx.ravel();yy=yy.ravel()
    rho=np.zeros((16,32,32));occ=np.zeros((2,16,32,32));zl=np.arange(16)[:,None]*50;zh=zl+50
    def clip(a,b):
        lo=np.maximum(zl,a+dz);hi=np.minimum(zh,b+dz);return lo,np.maximum(lo,hi)
    for iy in range(32):
        for ix in range(32):
            x=ix*50+xx-800-dx;y=iy*50+yy-800-dy;u=x*np.cos(theta)+y*np.sin(theta);v=-x*np.sin(theta)+y*np.cos(theta)
            a,b,c,d=intervals(g,u,v);lo1,hi1=clip(a,b);lo2,hi2=clip(c,d)
            first=integral(lo1,hi1,u,v,row['density'][0]);second=integral(lo2,hi2,u,v,row['density'][1]);vol1=hi1-lo1;vol2=hi2-lo2
            if g['family']==0:
                ol,oh=clip(np.maximum(a,c),np.minimum(b,d));second-=integral(ol,oh,u,v,row['density'][1]);vol2-=oh-ol
            rho[:,iy,ix]=(first+second).mean(axis=1)/50;occ[0,:,iy,ix]=vol1.mean(axis=1)/50;occ[1,:,iy,ix]=vol2.mean(axis=1)/50
    return rho,occ
