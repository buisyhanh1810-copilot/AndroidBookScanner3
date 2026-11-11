"""Ảnh sách mở giả lập có chân trị (ground truth) chính xác theo cm."""
import numpy as np, cv2, random

WORDS="the of and to in is that for it as was with be by on not he this are or his from at which but have an had they you were their one all we can her has there been if more when will would who so no".split()
PPC=56.0   # px/cm của ảnh phẳng

def rot(roll,pitch,yaw):
    r,p,y=np.deg2rad([roll,pitch,yaw])
    Rz=np.array([[np.cos(r),-np.sin(r),0],[np.sin(r),np.cos(r),0],[0,0,1]])
    Rx=np.array([[1,0,0],[0,np.cos(p),-np.sin(p)],[0,np.sin(p),np.cos(p)]])
    Ry=np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]])
    return Rz@Rx@Ry

def flat_spread(seed,Sw,Sh,m_in=1.5,m_out=2.0,m_top=2.2,m_bot=2.4,photo=False,blank=0.0,depth=0.55,blur_s=3.5,curl=0.18,light=0.12):
    rng=random.Random(seed)
    FW,FH=int(Sw*PPC),int(Sh*PPC)
    paper=np.array((226,231,235),np.uint8)
    img=np.tile(paper,(FH,FW,1)); ink=np.zeros((FH,FW),np.uint8)
    gx=FW//2
    def put(x,y,t,sc=0.7,th=2):
        cv2.putText(img,t,(x,y),cv2.FONT_HERSHEY_SIMPLEX,sc,(38,38,38),th,cv2.LINE_AA)
        cv2.putText(ink,t,(x,y),cv2.FONT_HERSHEY_SIMPLEX,sc,255,th)
    side=[]
    for si,(xa,xb) in enumerate([(int(m_out*PPC),gx-int(m_in*PPC)),(gx+int(m_in*PPC),FW-int(m_out*PPC))]):
        y0=int(m_top*PPC); y1=FH-int(m_bot*PPC)
        # đầu trang + số trang
        put(xa,int(1.3*PPC),"chapter "+rng.choice(WORDS),0.5,1)
        pn=str(rng.randint(10,399)); (tw,_),_=cv2.getTextSize(pn,cv2.FONT_HERSHEY_SIMPLEX,0.6,1)
        put((xa+xb)//2-tw//2,FH-int(1.1*PPC),pn,0.6,1)
        if photo and si==rng.randint(0,1):
            ph=(xb-xa)//2; pyh=int(5*PPC); px0=xa+(xb-xa)//4; py0=y0+int(6*PPC)
            yy,xx=np.mgrid[0:pyh,0:ph]
            g=(110+80*np.sin(xx/37.0)*np.cos(yy/29.0)+np.random.RandomState(seed).randn(pyh,ph)*10).astype(np.float32)
            img[py0:py0+pyh,px0:px0+ph]=np.clip(g,0,255)[...,None]
            ink[py0:py0+pyh,px0:px0+ph]=255
        y=y0
        yend=y1 if (si==0 or blank<=0) else y0+int((1-blank)*(y1-y0))
        while y<yend:
            if photo and abs(y-(y0+int(6*PPC)+int(2.5*PPC)))<int(3.3*PPC) and si==0 and False: y+=34; continue
            x=xa
            while x<xb-40:
                wd=rng.choice(WORDS)
                (tw,_),_=cv2.getTextSize(wd,cv2.FONT_HERSHEY_SIMPLEX,0.7,2)
                if x+tw>xb: break
                put(x,y,wd); x+=tw+14
            y+=34
    # ánh sáng không đều + bóng gáy + mờ gần gáy + cong
    X=np.arange(FW,dtype=np.float32)
    sh=1-depth*np.exp(-((X-gx)/(1.0*PPC))**2)-0.10*np.exp(-((X-gx)/(3.5*PPC))**2)
    gradl=1-light*(X/FW)
    yy=np.arange(FH,dtype=np.float32)[:,None]
    img=img.astype(np.float32)
    bl=cv2.GaussianBlur(img,(0,0),blur_s)
    wm=np.exp(-((X-gx)/(2.3*PPC))**2)[None,:,None]
    img=(img*(1-wm)+bl*wm)*sh[None,:,None]*gradl[None,:,None]*(1-0.05*yy/FH)[...,None]
    # mép ngoài: xấp giấy (các vạch mảnh sát mép) + tối nhẹ
    edge=np.minimum(X,FW-X)
    img=img*(1-0.12*np.exp(-edge/(0.4*PPC)))[None,:,None]
    for k in (4,9,15):
        img[:,k]*=0.93; img[:,FW-1-k]*=0.93
    img=np.clip(img,0,255).astype(np.uint8)
    # cong: nén ngang gần gáy (bề mặt cong khỏi mặt phẳng). Ánh xạ đích->nguồn
    if curl>0:
        xs=X.copy(); s=2.4*PPC
        dstx=xs-(gx)
        # đích d, nguồn u: d = (u-gx)*(1-curl*exp(-((u-gx)/s)^2)) (đơn điệu nếu curl<0.5)
        u=np.arange(FW,dtype=np.float32)-gx
        d=u*(1-curl*np.exp(-(u/s)**2))
        src=np.interp(dstx,d,u)+gx
        mapx=np.tile(src[None,:],(FH,1)).astype(np.float32); mapy=np.tile(np.arange(FH,dtype=np.float32)[:,None],(1,FW))
        img=cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        ink=cv2.remap(ink,mapx,mapy,cv2.INTER_NEAREST,borderMode=cv2.BORDER_REPLICATE)
    return img,ink,gx

def background(kind,Wc,Hc,seed):
    nr=np.random.RandomState(seed)
    if kind=='dark': bg=np.full((Hc,Wc,3),(35,38,42),np.float32)
    elif kind=='white': bg=np.full((Hc,Wc,3),(236,238,241),np.float32)
    elif kind=='wood':
        n=cv2.GaussianBlur(nr.randn(1,Wc).astype(np.float32),(0,0),5)*70
        bg=np.zeros((Hc,Wc,3),np.float32); bg[:]=(58,98,150); bg+=n[0][None,:,None]
    elif kind=='gray': bg=np.full((Hc,Wc,3),(150,150,152),np.float32)
    elif kind=='gradient':
        g=np.linspace(225,70,Wc,dtype=np.float32)[None,:,None]; bg=np.tile(g,(Hc,1,3))
    elif kind=='cloth':
        n=cv2.GaussianBlur(nr.randn(Hc,Wc).astype(np.float32),(0,0),2.0)*30
        bg=np.full((Hc,Wc,3),(120,150,170),np.float32)+n[...,None]
    return bg+nr.randn(Hc,Wc,1)*4

def make_case(seed,kind,Sw,Sh,pitch,yaw,roll,f_mult=1.0,edge_blur=1.8,photo=False,m_in=1.5,fill=0.72,Wc=2000,Hc=1500,noise=3.0,jpegq=85,shift=(0,0),**kw):
    img,ink,gx=flat_spread(seed,Sw,Sh,m_in=m_in,photo=photo,**kw)
    FH,FW=img.shape[:2]
    f=1500.0*f_mult
    R=rot(roll,pitch,yaw)
    d=f*Sw/(fill*Wc)
    K=np.array([[f,0,Wc/2],[0,f,Hc/2],[0,0,1]])
    A=np.array([[1/PPC,0,-FW/(2*PPC)],[0,1/PPC,-FH/(2*PPC)],[0,0,1]])
    K[0,2]+=shift[0]; K[1,2]+=shift[1]
    Hc3=K@np.column_stack([R[:,0],R[:,1],R@np.array([0,0,0])+np.array([0,0,d])])@A
    bg=background(kind,Wc,Hc,seed)
    pre=cv2.GaussianBlur(img,(0,0),0.6)
    wp=cv2.warpPerspective(pre,Hc3,(Wc,Hc),flags=cv2.INTER_LINEAR).astype(np.float32)
    mask=cv2.warpPerspective(np.full((FH,FW),255,np.uint8),Hc3,(Wc,Hc))
    maskf=cv2.GaussianBlur(mask,(0,0),edge_blur).astype(np.float32)[...,None]/255
    # bóng đổ xuống nền quanh sách
    cast=cv2.GaussianBlur(cv2.warpPerspective(np.full((FH,FW),255,np.uint8),Hc3+np.array([[0,0,14],[0,0,22],[0,0,0]]),(Wc,Hc)),(0,0),16).astype(np.float32)[...,None]/255
    bgs=bg*(1-0.35*cast*(1-maskf))
    out=wp*maskf+bgs*(1-maskf)
    out=out+np.random.RandomState(seed).randn(Hc,Wc,1)*noise
    out=np.clip(out,0,255).astype(np.uint8)
    ok,enc=cv2.imencode('.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,jpegq]); out=cv2.imdecode(enc,1)
    corners=np.float32([[0,0],[FW,0],[FW,FH],[0,FH]])
    qgt=cv2.perspectiveTransform(corners[None],Hc3)[0]
    ys,xs=np.nonzero(ink)
    if len(xs)>40000:
        idx=np.random.RandomState(1).choice(len(xs),40000,replace=False); ys,xs=ys[idx],xs[idx]
    return out,dict(H=Hc3,quad=qgt.astype(np.float32),ink=np.stack([xs,ys],1).astype(np.float32),gx=gx,FW=FW,FH=FH,Sw=Sw,Sh=Sh)
