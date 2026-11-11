import sys, numpy as np, cv2
from harness import *
import baseline, edge, edge2
def old(img,Sw,Sh):
    q=baseline.old_quad(img)
    if q is None: return None
    sp,M=baseline.warp_px(img,q); cx=baseline.old_gutter(sp)
    return q,M,sp.shape[1],sp.shape[0],cx
def new(img,Sw,Sh):
    q=edge.detect_quad(img)
    top=max(np.linalg.norm(q[0]-q[1]),np.linalg.norm(q[3]-q[2])); side=max(np.linalg.norm(q[0]-q[3]),np.linalg.norm(q[1]-q[2]))
    ppc=min((top/Sw+side/Sh)/2,300/2.54)
    W,H=int(round(Sw*ppc)),int(round(Sh*ppc))
    dst=np.float32([[0,0],[W-1,0],[W-1,H-1],[0,H-1]])
    M=cv2.getPerspectiveTransform(q.astype(np.float32),dst)
    sp=cv2.warpPerspective(img,M,(W,H),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
    cx,risky=edge.find_cut(sp); cx=int(np.clip(cx,.44*W,.56*W))
    return q,M,W,H,cx
def _cm(q,img,Sw,Sh,det=None):
    top=max(np.linalg.norm(q[0]-q[1]),np.linalg.norm(q[3]-q[2])); side=max(np.linalg.norm(q[0]-q[3]),np.linalg.norm(q[1]-q[2]))
    ppc=min((top/Sw+side/Sh)/2,300/2.54)
    W,H=int(round(Sw*ppc)),int(round(Sh*ppc))
    dst=np.float32([[0,0],[W-1,0],[W-1,H-1],[0,H-1]])
    M=cv2.getPerspectiveTransform(q.astype(np.float32),dst)
    sp=cv2.warpPerspective(img,M,(W,H),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
    cx,risky=edge.find_cut(sp); cx=int(np.clip(cx,.44*W,.56*W))
    return q,M,W,H,cx
def v2(img,Sw,Sh):
    q=edge2.detect_quad2(img,Sw,Sh)
    if q is None: return None
    return _cm(q,img,Sw,Sh)
def v2prior(img,Sw,Sh):
    q=edge2.detect_quad2(img,Sw,Sh,snap=False)
    if q is None: return None
    return _cm(q,img,Sw,Sh)
if __name__=='__main__':
    n=int(sys.argv[1]) if len(sys.argv)>1 else 36
    seed0=int(sys.argv[2]) if len(sys.argv)>2 else 100
    run({'old':old,'v2':v2,'prior':v2prior},n,verbose=False,seed0=seed0)
