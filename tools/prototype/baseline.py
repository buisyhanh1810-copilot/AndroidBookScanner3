"""Thuật toán CŨ của repo (Otsu + enclosingQuad + findGutter) viết lại bằng Python để đối chứng."""
import numpy as np, cv2

def order_points(p):
    p=np.array(p,np.float32); s=p.sum(1); d=p[:,1]-p[:,0]
    return np.array([p[s.argmin()],p[d.argmin()],p[s.argmax()],p[d.argmax()]],np.float32)

def enclosing(q,hull,maxshift):
    n=4; a=np.zeros(4);b=np.zeros(4);c=np.zeros(4)
    for i in range(4):
        p=q[i];r=q[(i+1)%4];dx,dy=r-p;ln=np.hypot(dx,dy)
        nx,ny=dy/ln,-dx/ln
        cc=nx*p[0]+ny*p[1]
        cc=max(cc,(hull[:,0]*nx+hull[:,1]*ny).max())
        a[i],b[i],c[i]=nx,ny,cc
    out=q.copy()
    for i in range(4):
        j=(i+3)%4; det=a[j]*b[i]-a[i]*b[j]
        if abs(det)<1e-6: continue
        x=(c[j]*b[i]-c[i]*b[j])/det; y=(a[j]*c[i]-a[i]*c[j])/det
        if np.hypot(x-q[i][0],y-q[i][1])<=maxshift: out[i]=(x,y)
    return out

def old_quad(bgr):
    scale=800.0/max(bgr.shape[:2])
    small=cv2.resize(bgr,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    g=cv2.GaussianBlur(cv2.cvtColor(small,cv2.COLOR_BGR2GRAY),(7,7),0)
    _,b=cv2.threshold(g,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    k=cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
    b=cv2.morphologyEx(b,cv2.MORPH_CLOSE,k);b=cv2.morphologyEx(b,cv2.MORPH_OPEN,k)
    cs,_=cv2.findContours(b,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    ia=small.shape[0]*small.shape[1];c0=(small.shape[1]/2,small.shape[0]/2)
    best=None;bs=0
    for c in cs:
        a=cv2.contourArea(c)
        if a<0.2*ia: continue
        ins=cv2.pointPolygonTest(c,c0,False)>=0
        s=a*(1 if ins else .3)
        if s>bs:bs=s;best=c
    if best is None: return None
    hull=cv2.convexHull(best);peri=cv2.arcLength(hull,True);quad=None;eps=.01
    while eps<=.12:
        ap=cv2.approxPolyDP(hull,eps*peri,True)
        if len(ap)==4:quad=ap.reshape(4,2).astype(np.float32);break
        if len(ap)<4:break
        eps+=.005
    if quad is None: quad=cv2.boxPoints(cv2.minAreaRect(hull))
    o=order_points(quad)
    ms=0.15*np.hypot(*small.shape[:2][::-1])
    enc=enclosing(o,hull.reshape(-1,2).astype(np.float64),ms)
    return order_points(enc/scale)

def old_gutter(spread):
    w=spread.shape[1]
    g=cv2.cvtColor(spread,cv2.COLOR_BGR2GRAY).astype(np.float32)
    col=g.mean(0); k=w//40; k=k+1 if k%2==0 else k
    col=cv2.GaussianBlur(col.reshape(1,-1),(max(k,5),1),0).ravel()
    lo,hi=int(w*.42),int(w*.58)
    bi=lo+int(col[lo:hi+1].argmin());bv=col[bi]
    med=np.sort(col)[w//2]
    return w//2 if bv>0.97*med else bi

def warp_px(bgr,q):
    tl,tr,br,bl=q
    w=int(max(np.hypot(*(tr-tl)),np.hypot(*(br-bl)))); h=int(max(np.hypot(*(bl-tl)),np.hypot(*(br-tr))))
    w=max(w,200);h=max(h,200)
    dst=np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    M=cv2.getPerspectiveTransform(q.astype(np.float32),dst)
    return cv2.warpPerspective(bgr,M,(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE),M
