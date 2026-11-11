import numpy as np, cv2, time, random, sys
from synth2 import *
import baseline

def metrics(gt,M,W,H,cx):
    """M: ảnh->đầu ra (3x3). W,H: kích thước đầu ra px. cx: vết cắt trong đầu ra."""
    Sw,Sh=gt['Sw'],gt['Sh']; FW,FH=gt['FW'],gt['FH']
    T=M@gt['H']
    # lưới điểm phẳng -> sai số kích thước (cm)
    gx_,gy_=np.meshgrid(np.linspace(0,FW,9),np.linspace(0,FH,7))
    P=np.stack([gx_.ravel(),gy_.ravel()],1).astype(np.float32)
    mp=cv2.perspectiveTransform(P[None],T)[0]
    ideal=np.stack([P[:,0]/FW*W,P[:,1]/FH*H],1)
    d=mp-ideal; dcm=np.hypot(d[:,0]/(W/Sw),d[:,1]/(H/Sh))
    ink=gt['ink']; o=cv2.perspectiveTransform(ink[None],T)[0]
    ppc=np.array([W/Sw,H/Sh])
    inside=(o[:,0]>=0)&(o[:,0]<=W)&(o[:,1]>=0)&(o[:,1]<=H)
    clipfrac=1-inside.mean()
    margin=min(o[:,0].min()/ppc[0],(W-o[:,0].max())/ppc[0],o[:,1].min()/ppc[1],(H-o[:,1].max())/ppc[1])
    left=ink[:,0]<gt['gx']
    leak=((o[~left,0]<cx).sum()+(o[left,0]>=cx).sum())/len(ink)*100
    clr=min(cx-o[left,0].max(),o[~left,0].min()-cx)/ppc[0]
    return dict(size_mean=dcm.mean()*10,size_max=dcm.max()*10,clip=clipfrac*100,margin=margin*10,leak=leak,clear=clr*10)

def gen_cases(n,seed0=100):
    rng=random.Random(seed0); out=[]
    kinds=['dark','white','wood','gray','gradient','cloth']
    cams=[(0,0,0),(12,4,3),(20,-8,-6),(28,9,6),(8,-12,2)]
    sizes=[(30,21),(42,29.7),(34,24)]
    for i in range(n):
        out.append(dict(seed=seed0+i,kind=kinds[i%len(kinds)],size=rng.choice(sizes),cam=rng.choice(cams),
            m_in=rng.choice([1.5,1.1,0.8]),m_out=rng.uniform(1.4,2.6),m_top=rng.uniform(1.8,2.8),m_bot=rng.uniform(2.0,3.0),photo=rng.random()<0.3,eb=rng.choice([1.2,2.0,3.0]),
            blur_s=rng.choice([2.5,3.5,6.0]),curl=rng.choice([0.08,0.18,0.28]),depth=rng.choice([0.3,0.55,0.75])))
    return out

def run(methods,n=36,verbose=True,seed0=100):
    cases=gen_cases(n,seed0); rows={m:[] for m in methods}
    for c in cases:
        Sw,Sh=c['size']; p,y,r=c['cam']
        img,gt=make_case(c['seed'],c['kind'],Sw,Sh,p,y,r,edge_blur=c['eb'],photo=c['photo'],m_in=c['m_in'],blur_s=c['blur_s'],curl=c['curl'],depth=c['depth'],m_out=c['m_out'],m_top=c['m_top'],m_bot=c['m_bot'])
        line=f"{c['seed']} {c['kind']:8s} {Sw:g}x{Sh:g} cam={c['cam']} min={c['m_in']} curl={c['curl']}"
        if verbose: print(line)
        for name,fn in methods.items():
            t=time.time()
            try: res=fn(img,Sw,Sh)
            except Exception as e: res=None; print('  EXC',name,repr(e))
            dt=time.time()-t
            if res is None: rows[name].append(None); print(f"  {name:6s} FAIL"); continue
            q,M,W,H,cx=res
            m=metrics(gt,M,W,H,cx)
            # sai số góc theo mm trên giấy
            Hinv=np.linalg.inv(gt['H']); qq=cv2.perspectiveTransform(q.reshape(1,4,2).astype(np.float32),Hinv.astype(np.float32))[0]
            cgt=np.float32([[0,0],[gt['FW'],0],[gt['FW'],gt['FH']],[0,gt['FH']]])
            m['corner']=np.max(np.hypot(*(((qq-cgt)/PPC*10).T)))
            m['t']=dt; rows[name].append(m)
            if verbose: print(f"  {name:6s} corner={m['corner']:6.1f}mm size_mean={m['size_mean']:6.1f} max={m['size_max']:6.1f}mm clip={m['clip']:5.2f}% margin={m['margin']:6.1f}mm leak={m['leak']:5.2f}% clear={m['clear']:6.1f}mm t={dt:.2f}s")
    print("\n=== TỔNG KẾT ===")
    for name,rs in rows.items():
        ok=[r for r in rs if r]
        f=lambda k:np.array([r[k] for r in ok])
        print(f"{name:6s} fail={len(rs)-len(ok)}/{len(rs)}  corner mean={f('corner').mean():.1f} p90={np.percentile(f('corner'),90):.1f} max={f('corner').max():.1f}mm | "
              f"size mean={f('size_mean').mean():.1f} max={f('size_max').max():.1f}mm | clip>0.1%: {(f('clip')>0.1).sum()} cases (max {f('clip').max():.2f}%) | "
              f"margin min={f('margin').min():.1f}mm | leak>0: {(f('leak')>0.0).sum()} cases (max {f('leak').max():.2f}%) | clear min={f('clear').min():.1f}mm | t={f('t').mean():.2f}s")
    return rows
