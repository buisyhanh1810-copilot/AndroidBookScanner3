"""Phát hiện viền sách v2: mốc là KHỐI CHỮ + lề điển hình theo kích thước sách đã biết,
rồi 'bắt dính' vào mép giấy thật bằng các tín hiệu sáng/tối, sắc độ, mép xấp giấy.
Mép không đủ tin cậy thì giữ vị trí dự đoán (không đoán bừa theo nền)."""
import numpy as np, cv2
from edge import text_mask, odd, order_points, find_cut

def _bil_maps(img,X,Y):
    return cv2.remap(img,X.astype(np.float32),Y.astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)

def text_points(L8,w,h,long):
    """Điểm nét chữ THÂN VĂN BẢN: nét chữ + mật độ cục bộ cao (loại vết mép giấy, bóng, chữ lẻ)."""
    tm=text_mask(L8)
    # chỉ giữ thành phần nhỏ cỡ chữ/từ; loại vệt dài mỏng (mép giấy, xấp giấy, bóng, cạnh bàn)
    n0,lab0,st0,_=cv2.connectedComponentsWithStats(tm,connectivity=8)
    ok0=np.zeros(n0,bool)
    for i in range(1,n0):
        cw,ch=st0[i,cv2.CC_STAT_WIDTH],st0[i,cv2.CC_STAT_HEIGHT]
        if cw<=0.07*long and ch<=0.035*long and st0[i,cv2.CC_STAT_AREA]>=3 and max(cw,ch)/max(1,min(cw,ch))<12:
            ok0[i]=True
    tm=(ok0[lab0]).astype(np.uint8)
    k=odd(max(15,0.03*long))
    dens=cv2.boxFilter(tm.astype(np.float32),-1,(k,k),normalize=True)
    core=(dens>0.10).astype(np.uint8)
    core=cv2.morphologyEx(core,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(odd(.03*w),odd(.02*h))))
    n,lab,stats,_=cv2.connectedComponentsWithStats(core)
    keep=np.zeros(n,bool)
    for i in range(1,n):
        if stats[i,cv2.CC_STAT_AREA]>=0.02*h*w: keep[i]=True
    ys,xs=np.nonzero((tm>0)&keep[lab])
    if len(xs)>8000:
        idx=np.random.RandomState(1).choice(len(xs),8000,replace=False); ys,xs=ys[idx],xs[idx]
    return np.stack([xs,ys],1).astype(np.float32)

def _support_line(tp,d0,o0,tol,span=25.0,step=0.5):
    """Đường tựa của khối chữ: quét góc, chọn góc cho NHIỀU điểm chữ thẳng hàng nhất sát biên ngoài
    (đầu dòng/cuối dòng thẳng hàng; bỏ qua đầu trang/số trang vì ít điểm)."""
    best=None
    for dg in np.arange(-span,span+1e-6,step):
        th=np.deg2rad(dg); c,sn=np.cos(th),np.sin(th)
        d=np.array([c*d0[0]-sn*d0[1],sn*d0[0]+c*d0[1]],np.float32)
        o=np.array([d[1],-d[0]],np.float32)
        if np.dot(o,o0)<0: o=-o
        po=tp@o
        lo,hi=np.percentile(po,78),np.percentile(po,99.95)
        if hi-lo<4*tol: continue
        nb=max(4,int((hi-lo)/tol))
        cnt,edges=np.histogram(po,bins=nb,range=(lo,hi))
        # cộng 2 bin kề nhau để chịu được lệch nhẹ do cong/nhiễu
        c2=cnt[:-1]+cnt[1:]
        j=int(c2.argmax()); sc=c2[j]-0.15*abs(dg)
        if best is None or sc>best[0]: best=(sc,d,o,edges[j],edges[j+2])
    if best is None: return None
    _,d,o,t0,t1=best
    # chọn dòng NGOÀI CÙNG có đủ điểm (>=55% đỉnh), không phải dòng đông nhất
    po=tp@o
    lo,hi=np.percentile(po,78),np.percentile(po,99.95)
    nb=max(4,int((hi-lo)/tol)); cnt,edges=np.histogram(po,bins=nb,range=(lo,hi)); c2=cnt[:-1]+cnt[1:]
    jo=int(np.nonzero(c2>=0.55*c2.max())[0].max()); t0,t1=edges[jo],edges[jo+2]
    pd=tp@d
    m=(po>=t0-0.5*tol)&(po<=t1+0.5*tol)
    pts=tp[m]
    if len(pts)>=6:
        vx,vy,x0,y0=cv2.fitLine(pts,cv2.DIST_HUBER,0,0.01,0.01).ravel()
        d2=np.array([vx,vy],np.float32)
        if np.dot(d2,d)<0: d2=-d2
        if abs(np.dot(d2,d))>np.cos(np.deg2rad(4)): d=d2
        o=np.array([d[1],-d[0]],np.float32)
        if np.dot(o,best[2])<0: o=-o
        p0=np.array([x0,y0],np.float32)
    else:
        p0=(t1*o+((pd.mean())*d)).astype(np.float32)
    # đặt đường tại biên ngoài của cụm (để chứa dòng chữ), không đẩy theo điểm lạc (tiêu đề ở ngoài)
    off=float(np.percentile((tp[m]-p0)@o,95)) if m.sum()>3 else 0.0
    return p0+o*off,d,o

def _envelope_fit(tp,d0,o0,nb=32,iters=4,qlo=10,qhi=90):
    """Bao ngoài khối chữ (chịu mép lởm chởm): điểm ngoài cùng theo dải (chỉ phần giữa của cạnh,
    tránh nhiễu ở góc), khớp Huber, lặp để bám đúng hướng nghiêng do phối cảnh."""
    d,o=d0.copy(),o0.copy(); res=None
    for _ in range(iters):
        pd=tp@d; po=tp@o
        lo,hi=np.percentile(pd,qlo),np.percentile(pd,qhi)
        edges=np.linspace(lo,hi,nb+1); pts=[]
        for k in range(nb):
            m=(pd>=edges[k])&(pd<edges[k+1])
            if m.sum()>=6: pts.append(((edges[k]+edges[k+1])/2)*d+np.percentile(po[m],98)*o)
        if len(pts)<8: return None
        vx,vy,x0,y0=cv2.fitLine(np.array(pts,np.float32),cv2.DIST_HUBER,0,0.01,0.01).ravel()
        d2=np.array([vx,vy],np.float32)
        if np.dot(d2,d)<0: d2=-d2
        d=d2; o=np.array([d[1],-d[0]],np.float32)
        if np.dot(o,o0)<0: o=-o
        res=(np.array([x0,y0],np.float32),d,o)
    p0,d,o=res
    off=float(np.percentile((tp-p0)@o,99.5)); return p0+o*off,d,o

def text_quad_lines_B(tp,long):
    """4 đường biên khối chữ [top,right,bottom,left] = (điểm, hướng, pháp tuyến ra ngoài).
    Hướng mỗi cạnh = trung bình (theo độ dài) các cạnh bao lồi có pháp tuyến gần hướng ban đầu
    (loại cạnh vát do góc thiếu chữ, vd trang cuối chương); vị trí = đường tựa ngoài cùng."""
    hull=cv2.convexHull(tp.reshape(-1,1,2)).reshape(-1,2).astype(np.float32)
    box=order_points(cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)); cen=hull.mean(0)
    n=len(hull); lines=[]
    ed=[(hull[j],hull[(j+1)%n]) for j in range(n)]
    for i in range(4):
        a,b=box[i],box[(i+1)%4]; d0=((b-a)/np.linalg.norm(b-a)).astype(np.float32)
        o0=np.array([d0[1],-d0[0]],np.float32)
        if np.dot(o0,(a+b)/2-box.mean(0))<0: o0=-o0
        d0=np.array([-o0[1],o0[0]],np.float32)          # d0 vuông góc o0, nhất quán
        num=0.0; den=0.0
        for p,q in ed:
            e=q-p; L=float(np.linalg.norm(e))
            if L<1e-3: continue
            e=e/L; nn=np.array([e[1],-e[0]],np.float32)
            if np.dot(nn,(p+q)/2-cen)<0: nn=-nn
            ang=np.arccos(np.clip(np.dot(nn,o0),-1,1))
            if ang<np.deg2rad(20):
                # góc lệch có dấu giữa e (căn theo d0) và d0
                if np.dot(e,d0)<0: e=-e
                th=np.arctan2(d0[0]*e[1]-d0[1]*e[0],np.dot(d0,e))
                num+=L*th; den+=L
        th=num/den if den>0 else 0.0
        c,sn=np.cos(th),np.sin(th)
        d=np.array([c*d0[0]-sn*d0[1],sn*d0[0]+c*d0[1]],np.float32)
        o=np.array([d[1],-d[0]],np.float32)
        if np.dot(o,o0)<0: o=-o
        off=float(np.percentile(tp@o,99.8))               # đường tựa ngoài cùng
        lines.append(((o*off).astype(np.float32),d,o))   # điểm trên đường: o*off (chân đường vuông góc từ gốc)
    return lines

def text_quad_lines_A(tp,long):
    """Bao lồi -> xấp xỉ 4 đỉnh -> đẩy mỗi cạnh ra ngoài tới khi chứa mọi điểm chữ."""
    hull=cv2.convexHull(tp.reshape(-1,1,2)).reshape(-1,2).astype(np.float32)
    peri=cv2.arcLength(hull.reshape(-1,1,2),True)
    quad=None; eps=0.01
    while eps<=0.15:
        ap=cv2.approxPolyDP(hull.reshape(-1,1,2),eps*peri,True)
        if len(ap)==4: quad=ap.reshape(4,2).astype(np.float32); break
        if len(ap)<4: break
        eps+=0.005
    if quad is None: quad=cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
    box=order_points(quad); cen=box.mean(0); lines=[]
    for i in range(4):
        a,b=box[i],box[(i+1)%4]; d=((b-a)/np.linalg.norm(b-a)).astype(np.float32)
        o=np.array([d[1],-d[0]],np.float32)
        if np.dot(o,(a+b)/2-cen)<0: o=-o
        off=float(np.percentile((tp-a)@o,99.8))
        lines.append((a+o*max(off,0.0),d,o))
    return lines

def _tightness(tp,line,long,nb=24):
    """Độ ôm sát của một đường biên: trung bình theo từng đoạn dọc cạnh của khoảng cách từ đường tới
    điểm chữ gần nhất trong đoạn đó (đường nghiêng sai hoặc cắt chéo góc thiếu chữ => lớn)."""
    p0,d,o=line
    pd=tp@d; dist=(p0@o)-(tp@o)
    lo,hi=np.percentile(pd,5),np.percentile(pd,95)
    edges=np.linspace(lo,hi,nb+1); v=[]
    cap=0.1*long
    for k in range(nb):
        m=(pd>=edges[k])&(pd<edges[k+1])
        if m.sum()>=3: v.append(min(float(np.maximum(dist[m],0).min()),cap))
    return float(np.mean(v)) if len(v)>=6 else 1e9

def _support_frac(tp,line,long,nb=24):
    """Tỉ lệ đoạn dọc cạnh có điểm chữ chạm sát đường biên (độ 'được đỡ' của cạnh)."""
    p0,d,o=line
    pd=tp@d; dist=(p0@o)-(tp@o)
    lo,hi=np.percentile(pd,2),np.percentile(pd,98)
    edges=np.linspace(lo,hi,nb+1); hit=0
    for k in range(nb):
        m=(pd>=edges[k])&(pd<edges[k+1])
        if m.sum()>=2 and float(np.maximum(dist[m],0).min())<0.014*long: hit+=1
    return hit/nb

def text_quad_lines(tp,long):
    """Dựng bằng hai cách (A: bao lồi 4 đỉnh, B: hướng cạnh bao lồi), mỗi CẠNH chọn đường ôm sát chữ hơn."""
    LA=LB=None
    try: LA=text_quad_lines_A(tp,long)
    except Exception: pass
    try: LB=text_quad_lines_B(tp,long)
    except Exception: pass
    cands=[L for L in (LA,LB) if L is not None]
    if not cands: return None
    if len(cands)==1: return cands[0]
    final=[]
    for i in range(4):
        sc=[_tightness(tp,L[i],long) for L in cands]
        final.append(cands[int(np.argmin(sc))][i])
    # cạnh ít được chữ đỡ (vd trang gần như trống) -> lấy hướng song song với cạnh đối diện
    sup=[_support_frac(tp,final[i],long) for i in range(4)]
    for i,j in ((0,2),(2,0),(1,3),(3,1)):
        ang=np.degrees(np.arccos(np.clip(abs(float(np.dot(final[i][1],final[j][1]))),0,1)))
        if sup[i]<0.45 and sup[j]>=0.6 and ang>14:      # chỉ khi hai cạnh đối diện lệch hướng bất thường (>14°)
            dj=final[j][1]; oi=final[i][2]
            di=dj if np.dot(dj,final[i][1])>0 else -dj
            oo=np.array([di[1],-di[0]],np.float32)
            if np.dot(oo,oi)<0: oo=-oo
            off=float(np.percentile(tp@oo,99.8))
            final[i]=((oo*off).astype(np.float32),di.astype(np.float32),oo)
    q=quad_from_lines(final)
    if q is not None and cv2.isContourConvex(q.reshape(-1,1,2).astype(np.float32)): return final
    return LA if LA is not None else LB

def inter(l1,l2):
    p1,d1=l1[0],l1[1]; p2,d2=l2[0],l2[1]
    A=np.array([d1,-d2]).T
    if abs(np.linalg.det(A))<1e-6: return None
    t=np.linalg.solve(A,p2-p1); return p1+d1*t[0]

def quad_from_lines(L):
    q=[inter(L[(i-1)%4],L[i]) for i in range(4)]   # góc i: giao cạnh i-1 và i
    if any(x is None for x in q): return None
    return order_points(np.array(q,np.float32))

def margin_prior(Sw,Sh):
    pw=Sw/2.0
    mo=0.095*pw; mi=0.075*pw; mt=0.060*Sh; mb=0.065*Sh
    tw=Sw-2*(mo+mi); th=Sh-(mt+mb)
    return mo,mi,mt,mb,tw,th

def detect_quad2(bgr,Sw,Sh,debug=None,snap=True):
    H0,W0=bgr.shape[:2]; sc=700.0/max(H0,W0)
    small=cv2.resize(bgr,None,fx=sc,fy=sc,interpolation=cv2.INTER_AREA)
    h,w=small.shape[:2]; long=max(h,w)
    lab=cv2.cvtColor(small,cv2.COLOR_BGR2Lab)
    L8=lab[:,:,0]; Lf=L8.astype(np.float32)
    tp=text_points(L8,w,h,long)
    if len(tp)<200: return None
    lines=text_quad_lines(tp,long)       # top,right,bottom,left
    if lines is None: return None
    TQ=quad_from_lines(lines)
    if TQ is None: return None
    mo,mi,mt,mb,tw,th=margin_prior(Sw,Sh)
    topw=(np.linalg.norm(TQ[1]-TQ[0])+np.linalg.norm(TQ[2]-TQ[3]))/2
    sideh=(np.linalg.norm(TQ[3]-TQ[0])+np.linalg.norm(TQ[2]-TQ[1]))/2
    # tỉ lệ px/cm: lấy theo cạnh nào ít nhạy với lề hơn (trung bình hình học)
    s=float(np.sqrt((topw/tw)*(sideh/th)))
    if not np.isfinite(s) or s<=1e-3: return None
    exp_m=[mt,mo,mb,mo]             # top,right,bottom,left
    prior=[]
    for (p,d,o),m in zip(lines,exp_m):
        prior.append((p+o*m*s,d,o))
    PQ=quad_from_lines(prior)
    if PQ is None: return None
    # ----- tín hiệu mép -----
    kc=odd(max(9,0.02*long))
    Lc=cv2.morphologyEx(Lf,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kc,kc)))
    Ls=cv2.GaussianBlur(Lc,(0,0),1.0)
    A=lab[:,:,1].astype(np.float32)-128; B=lab[:,:,2].astype(np.float32)-128
    cx0,cx1,cy0,cy1=int(.3*w),int(.7*w),int(.3*h),int(.7*h)
    P=float(np.percentile(Ls[cy0:cy1,cx0:cx1],85))
    sel=Ls[cy0:cy1,cx0:cx1]>=np.percentile(Ls[cy0:cy1,cx0:cx1],60)
    ma=float(np.median(A[cy0:cy1,cx0:cx1][sel])); mb_=float(np.median(B[cy0:cy1,cx0:cx1][sel]))
    chrom=cv2.GaussianBlur(np.exp(-(np.hypot(A-ma,B-mb_)/12.0)**2).astype(np.float32),(0,0),2.0)
    band=int(max(16,2.4*s)); K=28
    scales=[2,4,7,11]; gam=[1,.97,.93,.88]
    final=[]; dbg=[]
    sup=[_support_frac(tp,lines[i],long) for i in range(4)]
    for i,(pl,exp_) in enumerate(zip(prior,exp_m)):
        p0,d,o=pl
        band=int(max(16,2.4*s*(2.4 if sup[i]<0.45 else 1.0)))   # cạnh ít được chữ đỡ: tìm trong dải rộng hơn
        a=PQ[i]; b=PQ[(i+1)%4]
        # tia dọc theo cạnh dự đoán
        ts=0.1+0.8*np.arange(K)/(K-1)
        C=a[None,:]+(b-a)[None,:]*ts[:,None]
        us=np.arange(-band,band+1,dtype=np.float32)
        X=C[:,0:1]+us[None,:]*o[0]; Y=C[:,1:2]+us[None,:]*o[1]
        Lr=_bil_maps(Ls,X,Y); Cr=_bil_maps(chrom,X,Y)
        inb=(X>=0)&(X<=w-1)&(Y>=0)&(Y<=h-1)
        n=len(us)
        cl=np.concatenate([np.zeros((K,1)),np.cumsum(Lr,1)],1); cc=np.concatenate([np.zeros((K,1)),np.cumsum(Cr,1)],1)
        E=np.zeros((K,n)); Pc=np.zeros((K,n))
        for s_,g in zip(scales,gam):
            j=np.arange(s_,n-s_)
            lin=(cl[:,j]-cl[:,j-s_])/s_; lout=(cl[:,j+s_+1]-cl[:,j+1])/s_
            dl=(lin-lout)
            E[:,j]=np.maximum(E[:,j],g*(np.clip(dl,0,None)+0.6*np.clip(-dl,0,None)))
            cin=(cc[:,j]-cc[:,j-s_])/s_; cout=(cc[:,j+s_+1]-cc[:,j+1])/s_
            Pc[:,j]=np.maximum(Pc[:,j],g*np.clip(cin-cout,0,1))
        En=np.clip(E/max(float(np.percentile(E,97)),0.05*P),0,1)
        # cổng chữ: không cho biên cắt vào chữ (vị trí tính theo pháp tuyến)
        pos=us  # u=0 là vị trí dự đoán; chữ nằm ở u = -exp_*s
        ut=float(np.percentile(tp@o,99.7)-(p0@o))     # mép chữ thực tế (đơn vị u, 0 = vị trí dự đoán)
        gate=np.clip((us-ut-0.12*s)/(0.1*s),0.05,1.0)
        pri=np.exp(-0.5*(us/(0.6*s+1e-6)*0.9)**2)   # ưu tiên gần vị trí dự đoán
        S=gate[None,:]*(0.60*En+0.30*Pc+0.10*pri[None,:])
        S[~inb]=0; S[:,:6]=0; S[:,-6:]=0
        jb=S.argmax(1); smax=S[np.arange(K),jb]
        conf=np.zeros(K)
        for k in range(K):
            ex=S[k].copy(); ex[max(0,jb[k]-7):jb[k]+8]=-1
            s2=max(ex.max(),float(np.median(S[k])))
            conf[k]=np.clip((smax[k]-s2)/0.25,0,1)*np.clip(En[k,jb[k]]/0.35,0,1)
        u=us[jb].astype(np.float32)
        pts=C+o[None,:]*u[:,None]
        # RANSAC có trọng số
        tol=0.005*long; rng=np.random.RandomState(11+i)
        valid=np.nonzero(conf>0.2)[0]; bestin=None; bsc=0
        if snap and len(valid)>=5:
            sl=np.linalg.norm(b-a)
            for _ in range(120):
                p,q=rng.choice(valid,2,replace=False)
                dv=pts[q]-pts[p]; ln=np.linalg.norm(dv)
                if ln<0.25*sl: continue
                dv/=ln; nv=np.array([-dv[1],dv[0]])
                inl=(np.abs((pts-pts[p])@nv)<tol)&(conf>0.2)
                sc_=conf[inl].sum()
                if sc_>bsc: bsc=sc_;bestin=inl
        line=pl; snapped=False
        if bestin is not None and bestin.sum()>=5 and bsc/bestin.sum()>0.45 and bestin.sum()>=0.22*K:
            wgt=conf[bestin]; pp=pts[bestin]; mu=(pp*wgt[:,None]).sum(0)/wgt.sum()
            cov=((pp-mu).T*wgt)@(pp-mu); ev,evec=np.linalg.eigh(cov); dv=evec[:,1]
            if np.dot(dv,d)<0: dv=-dv
            o3=np.array([dv[1],-dv[0]],np.float32)
            if np.dot(o3,o)<0: o3=-o3
            # cạnh bắt dính chỉ chấp nhận nếu không lệch hướng quá 8° so với dự đoán
            if abs(np.dot(dv,d))>np.cos(np.deg2rad(22 if sup[i]<0.45 else 8)):
                line=(mu.astype(np.float32),dv.astype(np.float32),o3); snapped=True
        final.append(line); dbg.append((pts,conf,bestin,snapped,C))
    # an toàn: mọi nét chữ phải nằm trong quad (đệm 0.15 cm); đẩy cạnh ra nếu cần
    safe=[]
    for (p0,d,o) in final:
        need=float(np.percentile(tp@o,99.9)+0.15*s-(p0@o))
        safe.append((p0+o*need,d,o) if need>0 else (p0,d,o))
    final=safe
    q=quad_from_lines(final)
    ok=q is not None
    if ok:
        ar=cv2.contourArea(q); ap=cv2.contourArea(PQ)
        ok=cv2.isContourConvex(q.reshape(-1,1,2)) and 0.8*ap<ar<1.35*ap
    if not ok: q=PQ
    if debug is not None: debug.update(dict(TQ=TQ,PQ=PQ,dbg=dbg,sc=sc,s=s,tp=tp))
    q=np.clip(q,[0,0],[w-1,h-1])
    return order_points(q/sc)
