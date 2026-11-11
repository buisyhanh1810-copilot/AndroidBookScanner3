import sys, numpy as np, cv2
from harness import *
import pages
def evalcase(c):
    Sw,Sh=c['size']; p,y,r=c['cam']
    img,gt=make_case(c['seed'],c['kind'],Sw,Sh,p,y,r,edge_blur=c['eb'],photo=c['photo'],m_in=c['m_in'],blur_s=c['blur_s'],curl=c['curl'],depth=c['depth'],m_out=c['m_out'],m_top=c['m_top'],m_bot=c['m_bot'])
    res=pages.split_pages2(img,Sw,Sh)
    if res is None: return None
    outs,info,Ms,M1=res
    ppc=40.0; H=int(round(Sh*ppc)); pw=int(round(Sw/2*ppc))
    ink=gt['ink']; left=ink[:,0]<gt['gx']; R=[]
    for Mset in (M1,Ms):
        clip=0;n=0;marg=[]
        for side in (0,1):
            sel=left if side==0 else ~left
            o=cv2.perspectiveTransform(ink[sel][None].astype(np.float32),Mset[side]@gt['H'])[0]
            ins=(o[:,0]>=0)&(o[:,0]<=pw)&(o[:,1]>=0)&(o[:,1]<=H)
            clip+=(~ins).sum(); n+=len(o)
            marg.append(min(o[:,1].min(),H-o[:,1].max(),(o[:,0].min() if side==0 else pw-o[:,0].max()))/ppc*10)
        R.append((100*clip/n,min(marg)))
    return R
if __name__=='__main__':
    n=int(sys.argv[1]); seed=int(sys.argv[2]); a=[];b=[]
    for c in gen_cases(n,seed):
        try: r=evalcase(c)
        except Exception as e: print('ERR',c['seed'],e); continue
        if r: a.append(r[0]); b.append(r[1])
    for nm,v in (('stage1',a),('stage2',b)):
        v=np.array(v); print(nm,'clip>0.1%%: %d/%d max %.2f%%  margin min %.1fmm mean %.1fmm'%((v[:,0]>0.1).sum(),len(v),v[:,0].max(),v[:,1].min(),v[:,1].mean()))
