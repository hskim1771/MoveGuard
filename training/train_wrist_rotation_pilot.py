#!/usr/bin/env python3
"""Predict AI-Hub left/right wrist 3d_rot from MediaPipe landmarks."""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import cv2, mediapipe as mp, numpy as np, torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.linear_model import Ridge
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, n):
        super().__init__(); self.net=nn.Sequential(nn.Linear(n,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(.15),nn.Linear(256,128),nn.GELU(),nn.Linear(128,6))
    def forward(self,x): return self.net(x)


def scores(pred,y):
    delta=np.abs(pred-y); circ=np.minimum(delta,360-delta)
    return {'mae_all':float(circ.mean()),'rmse_all':float(np.sqrt((circ**2).mean())),
            'left_mae':float(circ[:,:3].mean()),'right_mae':float(circ[:,3:].mean()),
            'component_mae':{'left_roll':float(circ[:,0].mean()),'left_pitch':float(circ[:,1].mean()),'left_yaw':float(circ[:,2].mean()),
                             'right_roll':float(circ[:,3].mean()),'right_pitch':float(circ[:,4].mean()),'right_yaw':float(circ[:,5].mean())},
            'within_5':float((circ<=5).mean()),'within_10':float((circ<=10).mean()),'within_15':float((circ<=15).mean())}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base',type=Path,required=True); ap.add_argument('--model',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--seed',type=int,default=20260804); args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    manifest=json.loads((args.base/'labels_selected/selection_manifest.json').read_text()); rows=[]
    for r in manifest:
        if not r['label_3d']: continue
        d3=json.loads((args.base/r['label_3d']).read_text()); info=d3['info']; a=d3['annotations']
        rot=np.asarray([a['3d_rot'][20],a['3d_rot'][21]],np.float32).reshape(6)
        rows.append({'image':str(args.base/r['image']),'actor':info['actor_id'],'action':str(info['action_category_id']),'frame':int(a['frame_no']),'y':rot})
    groups=sorted({(r['actor'],r['action']) for r in rows}); random.Random(args.seed).shuffle(groups); n=len(groups); a=round(n*.70); b=a+round(n*.15)
    gs={'train':set(groups[:a]),'val':set(groups[a:b]),'test':set(groups[b:])}; split={k:[i for i,r in enumerate(rows) if (r['actor'],r['action']) in v] for k,v in gs.items()}
    (args.output/'wrist_rotation_split.json').write_text(json.dumps({'seed':args.seed,'groups':{k:[list(x) for x in sorted(v)] for k,v in gs.items()},'counts':{k:len(v) for k,v in split.items()}},indent=2))
    cache=args.output/'mediapipe_world_cache.npz'
    if cache.exists():
        c=np.load(cache); x,y,det=c['x'].astype(np.float32),c['y'].astype(np.float32),c['detected']
    else:
        opt=vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(args.model)),running_mode=vision.RunningMode.IMAGE,num_poses=1,min_pose_detection_confidence=.3,min_pose_presence_confidence=.3)
        detector=vision.PoseLandmarker.create_from_options(opt); xs=[]; ys=[]; ds=[]; started=time.time()
        for i,r in enumerate(rows,1):
            bgr=cv2.imread(r['image']); rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB); out=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)); ok=bool(out.pose_landmarks and out.pose_world_landmarks); ds.append(ok); ys.append(r['y'])
            if ok:
                p=np.asarray([[q.x,q.y,q.z,q.visibility] for q in out.pose_landmarks[0]],np.float32); w=np.asarray([[q.x,q.y,q.z] for q in out.pose_world_landmarks[0]],np.float32); feat=np.concatenate([p.reshape(-1),w.reshape(-1)])
            else: feat=np.zeros(231,np.float32)
            xs.append(feat)
            if i%250==0: print(json.dumps({'processed':i,'total':len(rows),'seconds':round(time.time()-started,1)}),flush=True)
        detector.close(); x=np.asarray(xs,np.float32); y=np.asarray(ys,np.float32); det=np.asarray(ds); np.savez_compressed(cache,x=x,y=y,detected=det)
    ids={k:[i for i in v if det[i]] for k,v in split.items()}; tr,va,te=ids['train'],ids['val'],ids['test']
    xm,xs=x[tr].mean(0),x[tr].std(0).clip(1e-5); ym,ys=y[tr].mean(0),y[tr].std(0).clip(1e-5); xn=(x-xm)/xs; yn=(y-ym)/ys
    mean_pred=np.repeat(ym[None],len(te),0)
    best_ridge=None
    for alpha in (.1,1.,10.,100.):
        m=Ridge(alpha=alpha).fit(xn[tr],y[tr]); mae=scores(m.predict(xn[va]),y[va])['mae_all']
        if best_ridge is None or mae<best_ridge[0]: best_ridge=(mae,alpha,m)
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu'); model=MLP(x.shape[1]).to(device); opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-4); loss=nn.SmoothL1Loss(beta=.2)
    def dl(idx,shuffle=False): return DataLoader(TensorDataset(torch.from_numpy(xn[idx].astype(np.float32)),torch.from_numpy(yn[idx].astype(np.float32))),batch_size=128,shuffle=shuffle)
    best=1e9; patience=0; hist=[]
    for epoch in range(100):
        model.train(); tl=[]
        for xb,yb in dl(tr,True): xb,yb=xb.to(device),yb.to(device); opt.zero_grad(); z=model(xb); l=loss(z,yb); l.backward(); opt.step(); tl.append(l.item())
        model.eval(); vl=[]
        with torch.no_grad():
            for xb,yb in dl(va): vl.append(loss(model(xb.to(device)),yb.to(device)).item())
        v=float(np.mean(vl)); hist.append({'epoch':epoch+1,'train_loss':float(np.mean(tl)),'val_loss':v})
        if v<best-1e-5: best=v; patience=0; torch.save({'state_dict':model.state_dict(),'x_mean':torch.from_numpy(xm),'x_std':torch.from_numpy(xs),'y_mean':torch.from_numpy(ym),'y_std':torch.from_numpy(ys)},args.output/'wrist_rotation_best.pt')
        else: patience+=1
        if patience>=15: break
    ck=torch.load(args.output/'wrist_rotation_best.pt',map_location=device); model.load_state_dict(ck['state_dict']); model.eval(); out=[]
    with torch.no_grad():
        for xb,_ in dl(te): out.append(model(xb.to(device)).cpu().numpy())
    mlp=np.concatenate(out)*ys+ym
    report={'seed':args.seed,'rotation_order_assumption':['roll','pitch','yaw'],'source_field':'annotations.3d_rot','rows':len(rows),'split_counts':{k:len(v) for k,v in split.items()},'detected_counts':{k:len(v) for k,v in ids.items()},'group_overlap':0,
            'target_range':{'min':y.min(0).tolist(),'max':y.max(0).tolist()},'mean_baseline':scores(mean_pred,y[te]),'ridge':{'alpha':best_ridge[1],**scores(best_ridge[2].predict(xn[te]),y[te])},'mlp':scores(mlp,y[te]),'best_val_loss':best,'epochs':hist}
    (args.output/'wrist_rotation_report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
