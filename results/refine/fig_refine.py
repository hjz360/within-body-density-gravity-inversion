import sys, numpy as np
sys.path.insert(0, '../best/v063figs')
from figstyle import plt, DENS0, save
B='/tmp/claude-0/-home-claude/0d44c39a-17a9-5017-8b98-70dc25555ded/scratchpad'
TH=B+'/best/theory12/'
G=np.load(B+'/repo_build/zenodo_data/G.npy').astype(np.float64)
T=np.load(TH+'理论真值与共同输入/density.npy').astype(float); L=np.load(TH+'理论真值与共同输入/body_labels.npy'); D=np.load(TH+'理论真值与共同输入/gravity.npy').astype(float)
P=np.load(TH+'pred_H_20260727_noise0.npy').astype(float)
cases=[(3,'水平分离双棱柱（渐变）'),(7,'倾斜板状体（渐变）')]
fig,ax=plt.subplots(2,5,figsize=(7.4,3.0),gridspec_kw=dict(width_ratios=[1,1,1,0.85,0.85],wspace=0.3,hspace=0.45))
for r,(i,nm) in enumerate(cases):
    R=np.load(f'ref_0.0_{i}.npy').astype(float); S=L[i]>0; jy=int(np.argmax(S.sum((0,2))))
    d=D[i].reshape(32,32); dn=(G@P[i].reshape(-1)).reshape(32,32); dr=(G@R.reshape(-1)).reshape(32,32)
    for j,(v,t) in enumerate([(T[i],'真值'),(P[i],'网络'),(R,'网络+正演修正')]):
        a=ax[r,j]; im=a.imshow(np.where(v[:,jy,:]<0.05,np.nan,v[:,jy,:]),cmap=DENS0,vmin=0,vmax=1.1,extent=(0,1600,800,0),aspect='auto',interpolation='nearest')
        a.contour(np.linspace(25,1575,32),np.linspace(25,775,16),S[:,jy,:].astype(float),levels=[0.5],colors='k',linewidths=0.6)
        a.set_title(f'{t}' if r==0 else '',fontsize=7.5); a.tick_params(labelsize=6)
        if j==0: a.set_ylabel(nm+'\n深度/m',fontsize=6.5)
        if j>0:
            p=v; pm,tm=p>.1,T[i]>0; dice=2*(pm&tm).sum()/(pm.sum()+tm.sum()); rr=np.linalg.norm((G@p.reshape(-1))-D[i].reshape(-1))/np.linalg.norm(D[i].reshape(-1))
            a.text(0.02,0.04,f'Dice {dice:.2f}  残差 {100*rr:.0f}%',transform=a.transAxes,fontsize=5.8,bbox=dict(fc='white',ec='none',pad=1))
    vm=0.12*d.max()
    for j,(dd,t) in enumerate([(dn-d,'网络残差'),(dr-d,'修正后残差')]):
        a=ax[r,3+j]; im2=a.imshow(dd,cmap='RdBu_r',vmin=-vm,vmax=vm,origin='lower',extent=(0,1600,0,1600)); a.set_title(t if r==0 else '',fontsize=7.5); a.tick_params(labelsize=5.5)
cb=fig.colorbar(im2,ax=ax[:,3:],shrink=0.7,pad=0.02); cb.set_label('预测减观测/mGal（±12%峰值）',fontsize=6); cb.ax.tick_params(labelsize=5.5)
save(fig,'refine_demo',out='.')
