import numpy as np, cv2, sys, random
from harness import *
import run2
rng=random.Random(9)
kinds=['dark','white','wood','gray','gradient','cloth']
S=[]  # (tên, kwargs)
base=dict(m_out=2.0,m_top=2.3,m_bot=2.5)
for i,(name,extra) in enumerate([
 ('góc rộng f×0.55',dict(f_mult=0.55,cam=(15,5,3))),('góc rộng f×0.55 nghiêng',dict(f_mult=0.55,cam=(25,-8,4))),
 ('tele f×1.8',dict(f_mult=1.8,cam=(15,5,3))),('tele f×1.8 nghiêng',dict(f_mult=1.8,cam=(28,9,6))),
 ('nhiễu σ9 + jpeg q55',dict(noise=9,jpegq=55,cam=(10,3,2))),('nhiễu σ9 + jpeg q55 nghiêng',dict(noise=9,jpegq=55,cam=(22,-8,-5))),
 ('1 trang chỉ nửa chữ',dict(blank=0.5,cam=(10,3,2))),('1 trang gần như trống',dict(blank=0.8,cam=(20,-6,3))),
 ('sách rất to (fill .92)',dict(fill=0.92,cam=(8,3,1))),('sách nhỏ (fill .45)',dict(fill=0.45,cam=(12,4,2))),
 ('lệch khỏi tâm khung',dict(shift=(240,160),fill=0.7,cam=(10,3,2))),('lệch tâm ngược',dict(shift=(-220,-140),fill=0.7,cam=(15,-5,-3))),
 ('không bóng gáy',dict(depth=0.05,blur_s=1.0,curl=0.0,cam=(12,4,2))),('gáy rất sâu + cong',dict(depth=0.85,curl=0.35,blur_s=7.0,cam=(12,4,2))),
 ('xoay roll 18°',dict(cam=(10,3,18))),('xoay roll -20°',dict(cam=(18,-6,-20))),
 ('có ảnh minh hoạ',dict(photo=True,cam=(14,4,3))),('cạnh mờ mạnh',dict(edge_blur=5.0,cam=(14,4,3))),
]):
    for k,kind in enumerate(['dark','white','gradient'] if 'rất to' not in name else ['dark','wood']):
        S.append((name,kind,extra))
rows=[]
for j,(name,kind,ex) in enumerate(S):
    ex=dict(ex); cam=ex.pop('cam'); Sw,Sh=rng.choice([(30,21),(34,24),(42,29.7)])
    kw=dict(base); kw.update(ex)
    img,gt=make_case(2000+j,kind,Sw,Sh,*cam,m_in=rng.choice([0.9,1.5]),**kw)
    try: res=run2.v2(img,Sw,Sh)
    except Exception as e: res=None; print('EXC',name,kind,repr(e))
    if res is None: rows.append((name,kind,None)); print(f"{name:28s} {kind:8s} FAIL"); continue
    q,M,W,H,cx=res; m=metrics(gt,M,W,H,cx)
    Hinv=np.linalg.inv(gt['H']).astype(np.float32); qq=cv2.perspectiveTransform(q.reshape(1,4,2).astype(np.float32),Hinv)[0]/PPC
    cg=np.float32([[0,0],[Sw,0],[Sw,Sh],[0,Sh]]); err=np.hypot(*(qq-cg).T)*10
    rows.append((name,kind,err.max(),m))
    print(f"{name:28s} {kind:8s} corner={err.max():5.1f}mm size_max={m['size_max']:5.1f}mm clip={m['clip']:5.2f}% leak={m['leak']:5.2f}% clear={m['clear']:5.1f}mm")
ok=[r for r in rows if r[2] is not None]
print("\nFAIL:",len(rows)-len(ok),"/",len(rows)," corner mean=%.1f max=%.1f"%(np.mean([r[2] for r in ok]),max(r[2] for r in ok)),
 "| clip>0.1%:",sum(r[3]['clip']>0.1 for r in ok),"| leak>0:",sum(r[3]['leak']>0 for r in ok),"| clear min=%.1f"%min(r[3]['clear'] for r in ok))
