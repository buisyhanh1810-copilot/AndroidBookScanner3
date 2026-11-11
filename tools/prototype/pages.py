"""Tách từng trang bằng homography riêng: mỗi trang là tứ giác (mép ngoài + đường gáy) -> chữ nhật."""
import numpy as np, cv2
import edge, edge2

def split_pages(im, Sw, Sh, ppc=40.0):
    q=edge2.detect_quad2(im,Sw,Sh)
    if q is None: return None
    W,H=int(round(Sw*ppc)),int(round(Sh*ppc))
    dst=np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]],np.float32)
    M=cv2.getPerspectiveTransform(q.astype(np.float32),dst); Mi=np.linalg.inv(M)
    sp=cv2.warpPerspective(im,M,(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    h2=H//2
    xt,_=edge.find_cut(sp[:h2]); xb,_=edge.find_cut(sp[h2:])
    lo,hi=0.44*W,0.56*W
    xt=float(np.clip(xt,lo,hi)); xb=float(np.clip(xb,lo,hi))
    if abs(xt-xb)>0.04*W: xt=xb=(xt+xb)/2        # gáy gần thẳng đứng; chênh lớn = nhiễu
    # đường gáy (qua (xt,H/4) và (xb,3H/4)) kéo dài tới mép trên/dưới
    k=(xb-xt)/(H/2.0)
    gt=np.array([xt-k*H/4,0.0]); gb=np.array([xb+k*H/4,H-1.0])
    pt=lambda p:cv2.perspectiveTransform(np.array([[p]],np.float32),Mi)[0,0]
    G_t,G_b=pt(gt),pt(gb)
    ql=np.array([q[0],G_t,G_b,q[3]],np.float32); qr=np.array([G_t,q[1],q[2],G_b],np.float32)
    pw=int(round(Sw/2*ppc))
    d2=np.array([[0,0],[pw-1,0],[pw-1,H-1],[0,H-1]],np.float32)
    out=[cv2.warpPerspective(im,cv2.getPerspectiveTransform(Q,d2),(pw,H),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE) for Q in (ql,qr)]
    return out,(q,G_t,G_b)

import edge3
def split_pages2(im, Sw, Sh, ppc=40.0, m_cm=1.5, g_cm=0.2):
    """Bước 1: cắt thô từng trang KÈM NỀN. Bước 2: mỗi trang dò lại 3 mép ngoài rồi nắn về chữ nhật."""
    r=split_pages(im,Sw,Sh,ppc)
    if r is None: return None
    _,(q,gt,gb)=r
    H=int(round(Sh*ppc)); pw=int(round(Sw/2*ppc)); m=int(round(m_cm*ppc)); g=int(round(g_cm*ppc))
    quads=[np.array([q[0],gt,gb,q[3]],np.float32), np.array([gt,q[1],q[2],gb],np.float32)]
    outs=[]; info=[]; Ms=[]; M1=[]
    for side,Q in enumerate(quads):
        x0=m if side==0 else g
        cw=m+pw+g; ch=H+2*m; y0=m
        dst=np.array([[x0,y0],[x0+pw-1,y0],[x0+pw-1,y0+H-1],[x0,y0+H-1]],np.float32)
        M=cv2.getPerspectiveTransform(Q,dst); Mi=np.linalg.inv(M)
        crop=cv2.warpPerspective(im,M,(cw,ch),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        outer=(0,2,3) if side==0 else (0,1,2)
        q2=edge3.refine_page(crop,(x0,y0,x0+pw,y0+H),ppc,outer)
        if q2 is None: q2=dst
        q2o=cv2.perspectiveTransform(q2.reshape(1,4,2).astype(np.float32),Mi)[0]
        d2=np.array([[0,0],[pw-1,0],[pw-1,H-1],[0,H-1]],np.float32)
        out=cv2.warpPerspective(im,cv2.getPerspectiveTransform(q2o.astype(np.float32),d2),(pw,H),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
        outs.append(out); info.append((crop,q2,dst)); Ms.append(cv2.getPerspectiveTransform(q2o.astype(np.float32),d2)); M1.append(cv2.getPerspectiveTransform(Q,d2))
    return outs,info,Ms,M1


def clean_bw(p):
    g=cv2.cvtColor(p,cv2.COLOR_BGR2GRAY)
    bg=cv2.GaussianBlur(cv2.dilate(g,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(41,41))),(0,0),21)
    f=np.clip(g.astype(np.float32)/np.maximum(bg,1)*255,0,255).astype(np.uint8)
    return cv2.threshold(f,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]

def process_and_merge(im,Sw,Sh,ppc=60.0):
    """Xử lý riêng từng trang rồi ghép lại thành 1 ảnh 2 trang."""
    outs=split_pages2(im,Sw,Sh,ppc)[0]
    cl=[clean_bw(p) for p in outs]
    hh=max(c.shape[0] for c in cl)
    pad=lambda c:cv2.copyMakeBorder(c,(hh-c.shape[0])//2,hh-c.shape[0]-(hh-c.shape[0])//2,0,0,cv2.BORDER_CONSTANT,value=255)
    return np.hstack([pad(c) for c in cl])
