import numpy as np, cv2, sys
from harness import *
import run2, edge2
seed0=int(sys.argv[1]); n=int(sys.argv[2])
rows=[]
for c in gen_cases(n,seed0):
    Sw,Sh=c['size']; p,y,r=c['cam']
    img,gt=make_case(c['seed'],c['kind'],Sw,Sh,p,y,r,edge_blur=c['eb'],photo=c['photo'],m_in=c['m_in'],blur_s=c['blur_s'],curl=c['curl'],depth=c['depth'],m_out=c['m_out'],m_top=c['m_top'],m_bot=c['m_bot'])
    res=run2.v2(img,Sw,Sh)
    q,M,W,H,cx=res; m=metrics(gt,M,W,H,cx)
    Hinv=np.linalg.inv(gt['H']).astype(np.float32)
    qq=cv2.perspectiveTransform(q.reshape(1,4,2).astype(np.float32),Hinv)[0]/PPC
    cg=np.float32([[0,0],[Sw,0],[Sw,Sh],[0,Sh]]); err=np.hypot(*(qq-cg).T)*10
    dbg={}; edge2.detect_quad2(img,Sw,Sh,debug=dbg)
    rows.append((err.max(),m['clear'],c['seed'],c['kind'],c['cam'],f"{Sw}x{Sh}",c['m_in'],c['curl'],c['photo'],round(c['depth'],2),round(c['blur_s'],1),[d[3] for d in dbg['dbg']],np.round(err).astype(int).tolist()))
print("--- sai số góc lớn nhất"); 
for r in sorted(rows,key=lambda r:-r[0])[:5]: print(r)
print("--- đệm vết cắt gáy nhỏ nhất (mm)")
for r in sorted(rows,key=lambda r:r[1])[:6]: print(round(r[1],1),r[2:9])
