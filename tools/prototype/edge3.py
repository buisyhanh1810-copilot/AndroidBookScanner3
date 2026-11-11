"""Bước 2 (từng trang): trang đã cắt thô KÈM NỀN -> dò lại 3 mép ngoài (trên, dưới, mép ngoài) -> homography về chữ nhật."""
import numpy as np, cv2
from edge import odd, order_points
import edge2
from edge2 import _bil_maps, _support_frac, text_points, inter, quad_from_lines

def snap_sides(small, tp, prior, PQ, s, sup, snap_ids, snap=True):
    """prior: 4 đường (p,d,o) [top,right,bottom,left]; chỉ các cạnh trong snap_ids được bắt dính vào mép thật."""
    h,w=small.shape[:2]; long=max(h,w)
    lab=cv2.cvtColor(small,cv2.COLOR_BGR2Lab); Lf=lab[:,:,0].astype(np.float32)
    kc=odd(max(9,0.02*long))
    Lc=cv2.morphologyEx(Lf,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kc,kc)))
    Ls=cv2.GaussianBlur(Lc,(0,0),1.0)
    A=lab[:,:,1].astype(np.float32)-128; B=lab[:,:,2].astype(np.float32)-128
    cx0,cx1,cy0,cy1=int(.3*w),int(.7*w),int(.3*h),int(.7*h)
    P=float(np.percentile(Ls[cy0:cy1,cx0:cx1],85))
    sel=Ls[cy0:cy1,cx0:cx1]>=np.percentile(Ls[cy0:cy1,cx0:cx1],60)
    ma=float(np.median(A[cy0:cy1,cx0:cx1][sel])); mb_=float(np.median(B[cy0:cy1,cx0:cx1][sel]))
    chrom=cv2.GaussianBlur(np.exp(-(np.hypot(A-ma,B-mb_)/12.0)**2).astype(np.float32),(0,0),2.0)
    K=28; scales=[2,4,7,11]; gam=[1,.97,.93,.88]; final=[]
    for i,pl in enumerate(prior):
        p0,d,o=pl
        if i not in snap_ids: final.append(pl); continue
        band=int(max(16,2.4*s*(2.4 if sup[i]<0.45 else 1.0)))
        a=PQ[i]; b=PQ[(i+1)%4]
        ts=0.1+0.8*np.arange(K)/(K-1); C=a[None,:]+(b-a)[None,:]*ts[:,None]
        us=np.arange(-band,band+1,dtype=np.float32)
        X=C[:,0:1]+us[None,:]*o[0]; Y=C[:,1:2]+us[None,:]*o[1]
        Lr=_bil_maps(Ls,X,Y); Cr=_bil_maps(chrom,X,Y)
        inb=(X>=0)&(X<=w-1)&(Y>=0)&(Y<=h-1); n=len(us)
        cl=np.concatenate([np.zeros((K,1)),np.cumsum(Lr,1)],1); cc=np.concatenate([np.zeros((K,1)),np.cumsum(Cr,1)],1)
        E=np.zeros((K,n)); Pc=np.zeros((K,n))
        for s_,g in zip(scales,gam):
            j=np.arange(s_,n-s_)
            lin=(cl[:,j]-cl[:,j-s_])/s_; lout=(cl[:,j+s_+1]-cl[:,j+1])/s_; dl=lin-lout
            E[:,j]=np.maximum(E[:,j],g*(np.clip(dl,0,None)+0.6*np.clip(-dl,0,None)))
            cin=(cc[:,j]-cc[:,j-s_])/s_; cout=(cc[:,j+s_+1]-cc[:,j+1])/s_
            Pc[:,j]=np.maximum(Pc[:,j],g*np.clip(cin-cout,0,1))
        En=np.clip(E/max(float(np.percentile(E,97)),0.05*P),0,1)
        ut=float(np.percentile(tp@o,99.7)-(p0@o))
        gate=np.clip((us-ut-0.12*s)/(0.1*s),0.05,1.0)
        pri=np.exp(-0.5*(us/(0.6*s+1e-6)*0.9)**2)
        S=gate[None,:]*(0.60*En+0.30*Pc+0.10*pri[None,:]); S[~inb]=0; S[:,:6]=0; S[:,-6:]=0
        jb=S.argmax(1); smax=S[np.arange(K),jb]; conf=np.zeros(K)
        for k in range(K):
            ex=S[k].copy(); ex[max(0,jb[k]-7):jb[k]+8]=-1
            conf[k]=np.clip((smax[k]-max(ex.max(),float(np.median(S[k]))))/0.25,0,1)*np.clip(En[k,jb[k]]/0.35,0,1)
        pts=C+o[None,:]*us[jb].astype(np.float32)[:,None]
        tol=0.005*long; rng=np.random.RandomState(11+i)
        valid=np.nonzero(conf>0.2)[0]; bestin=None; bsc=0
        if snap and len(valid)>=5:
            sl=np.linalg.norm(b-a)
            for _ in range(120):
                p,q=rng.choice(valid,2,replace=False); dv=pts[q]-pts[p]; ln=np.linalg.norm(dv)
                if ln<0.25*sl: continue
                dv/=ln; nv=np.array([-dv[1],dv[0]])
                inl=(np.abs((pts-pts[p])@nv)<tol)&(conf>0.2); sc_=conf[inl].sum()
                if sc_>bsc: bsc=sc_; bestin=inl
        line=pl
        if bestin is not None and bestin.sum()>=5 and bsc/bestin.sum()>0.45 and bestin.sum()>=0.22*K:
            wgt=conf[bestin]; pp=pts[bestin]; mu=(pp*wgt[:,None]).sum(0)/wgt.sum()
            cov=((pp-mu).T*wgt)@(pp-mu); ev,evec=np.linalg.eigh(cov); dv=evec[:,1]
            if np.dot(dv,d)<0: dv=-dv
            o3=np.array([dv[1],-dv[0]],np.float32)
            if np.dot(o3,o)<0: o3=-o3
            if abs(np.dot(dv,d))>np.cos(np.deg2rad(8)): line=(mu.astype(np.float32),dv.astype(np.float32),o3)
        final.append(line)
    return final

def refine_page(crop, inner, ppc, outer_sides, margin_cm=0.15):
    """crop: ảnh trang cắt thô kèm nền. inner=(x0,y0,x1,y1) toạ độ trang dự kiến trong crop (px).
    outer_sides: các cạnh cần dò ({0:top,1:right,2:bottom,3:left}). Trả về 4 góc [tl,tr,br,bl] trong crop."""
    H0,W0=crop.shape[:2]; sc=700.0/max(H0,W0)
    small=cv2.resize(crop,None,fx=sc,fy=sc,interpolation=cv2.INTER_AREA); h,w=small.shape[:2]
    x0,y0,x1,y1=[v*sc for v in inner]
    L8=cv2.cvtColor(small,cv2.COLOR_BGR2Lab)[:,:,0]
    tp=text_points(L8,w,h,max(h,w))
    if len(tp)<100: return None
    # tiên nghiệm: hình chữ nhật trang dự kiến (top,right,bottom,left; pháp tuyến ra ngoài)
    f=np.float32
    prior=[(np.array([0,y0],f),np.array([1,0],f),np.array([0,-1],f)),
           (np.array([x1,0],f),np.array([0,1],f),np.array([1,0],f)),
           (np.array([0,y1],f),np.array([1,0],f),np.array([0,1],f)),
           (np.array([x0,0],f),np.array([0,1],f),np.array([-1,0],f))]
    PQ=quad_from_lines(prior); s=ppc*sc
    sup=[1.0]*4
    final=snap_sides(small,tp,prior,PQ,s,sup,set(outer_sides))
    # an toàn: nét chữ nằm trong quad
    safe=[]
    for (p0,d,o) in final:
        need=float(np.percentile(tp@o,99.9)+margin_cm*s-(p0@o))
        safe.append((p0+o*need,d,o) if need>0 else (p0,d,o))
    q=quad_from_lines(safe)
    if q is None or not cv2.isContourConvex(q.reshape(-1,1,2)): q=PQ
    ar=cv2.contourArea(q); ap=cv2.contourArea(PQ)
    if not (0.8*ap<ar<1.25*ap): q=PQ
    return order_points(q/sc)
