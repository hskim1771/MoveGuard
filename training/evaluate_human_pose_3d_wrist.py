#!/usr/bin/env python3
"""Train/evaluate full-skeleton 3D regression with wrist-focused reporting."""
from __future__ import annotations
import argparse,csv,json,math,random,time
from collections import Counter,defaultdict
from pathlib import Path
import cv2,numpy as np,torch
from sklearn.linear_model import Ridge
from torch import nn
from torch.utils.data import DataLoader,TensorDataset

JOINTS=("Pelvis","L_Hip","R_Hip","Spine1","L_Knee","R_Knee","Spine2","L_Ankle","R_Ankle","Spine3","L_Foot","R_Foot","Neck","L_Collar","R_Collar","Head","L_Shoulder","R_Shoulder","L_Elbow","R_Elbow","L_Wrist","R_Wrist","L_Hand","R_Hand")
LW,RW=20,21

class MLP(nn.Module):
    def __init__(self,n):
        super().__init__(); self.net=nn.Sequential(nn.Linear(n,384),nn.LayerNorm(384),nn.GELU(),nn.Dropout(.15),nn.Linear(384,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,72))
    def forward(self,x): return self.net(x).reshape(-1,24,3)

def mp_skeleton(feat):
    w=feat[132:].reshape(33,3); pelvis=w[[23,24]].mean(0); neck=w[[11,12]].mean(0)
    return np.asarray([pelvis,w[23],w[24],pelvis*.75+neck*.25,w[25],w[26],(pelvis+neck)/2,w[27],w[28],pelvis*.25+neck*.75,w[31],w[32],neck,(neck+w[11])/2,(neck+w[12])/2,w[0],w[11],w[12],w[13],w[14],w[15],w[16],w[[17,19,21]].mean(0),w[[18,20,22]].mean(0)],np.float32)

def procrustes(pred,gt):
    pc=pred-pred.mean(0); gc=gt-gt.mean(0); u,s,vt=np.linalg.svd(pc.T@gc); r=u@vt
    if np.linalg.det(r)<0: vt[-1]*=-1; r=u@vt
    scale=s.sum()/(pc*pc).sum().clip(1e-9); return pc@r*scale+gt.mean(0)

def metric(pred,gt):
    d=np.abs(pred-gt); e=np.linalg.norm(pred-gt,axis=2)
    return {'left_mpjpe':float(e[:,LW].mean()),'right_mpjpe':float(e[:,RW].mean()),'wrist_mean_mpjpe':float(e[:,[LW,RW]].mean()),
            'all_mpjpe':float(e.mean()),'left_axis_mae':dict(zip('xyz',map(float,d[:,LW].mean(0)))),'right_axis_mae':dict(zip('xyz',map(float,d[:,RW].mean(0))))}

def pck2(pred,gt):
    span=gt.max(1)-gt.min(1); diag=np.linalg.norm(span,axis=1).clip(1e-6); e=np.linalg.norm(pred[:,[LW,RW]]-gt[:,[LW,RW]],axis=2)/diag[:,None]
    return {'pck_0_05':float((e<=.05).mean()),'pck_0_10':float((e<=.10).mean())}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base',type=Path,required=True); ap.add_argument('--old-checkpoint',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--results',type=Path,required=True); ap.add_argument('--seed',type=int,default=20260804); args=ap.parse_args(); args.checkpoint.mkdir(parents=True,exist_ok=True); args.results.mkdir(parents=True,exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    oldsplit=json.loads((args.old_checkpoint/'split_manifest.json').read_text()); gs={k:{tuple(x) for x in v} for k,v in oldsplit['groups'].items()}
    manifest=json.loads((args.base/'labels_selected/selection_manifest.json').read_text()); rows=[]
    for r in manifest:
        if not r['label_3d']: continue
        d2=json.loads((args.base/r['label_2d']).read_text()); d3=json.loads((args.base/r['label_3d']).read_text()); i,a=d3['info'],d3['annotations']; p=np.asarray(a['3d_pos'],np.float32).reshape(24,4)[:,:3]
        rows.append({'image':str(args.base/r['image']),'label2':str(args.base/r['label_2d']),'actor':i['actor_id'],'action':str(i['action_category_id']),'camera':int(d2['info']['camera_no']),'frame':int(a['frame_no']),'gt3':p,'gt2':np.asarray(d2['annotations']['2d_pos'],np.float32)/np.asarray([d2['info']['img_width'],d2['info']['img_height']],np.float32)})
    split={k:[i for i,r in enumerate(rows) if (r['action'],r['actor']) in v or (r['actor'],r['action']) in v] for k,v in gs.items()}
    cache=np.load(args.old_checkpoint/'mediapipe_world_cache.npz'); x=cache['x'].astype(np.float32); detected=cache['detected']; assert len(x)==len(rows)
    ids={k:[i for i in v if detected[i]] for k,v in split.items()}; tr,va,te=ids['train'],ids['val'],ids['test']; y=np.asarray([r['gt3'] for r in rows],np.float32)
    xm,xs=x[tr].mean(0),x[tr].std(0).clip(1e-5); ym,ys=y[tr].mean((0,1)),y[tr].std((0,1)).clip(1e-5); xn=(x-xm)/xs; yn=(y-ym)/ys
    ridge_best=None
    for alpha in (1.,10.,100.,1000.):
        m=Ridge(alpha=alpha).fit(xn[tr],yn[tr].reshape(len(tr),-1)); v=m.predict(xn[va]).reshape(-1,24,3)*ys+ym; score=metric(v,y[va])['wrist_mean_mpjpe']
        if ridge_best is None or score<ridge_best[0]: ridge_best=(score,alpha,m)
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu'); model=MLP(x.shape[1]).to(device); opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-4); loss=nn.SmoothL1Loss(beta=.15)
    def dl(idx,shuffle=False): return DataLoader(TensorDataset(torch.from_numpy(xn[idx].astype(np.float32)),torch.from_numpy(yn[idx].astype(np.float32))),batch_size=128,shuffle=shuffle)
    best=1e9; patience=0; hist=[]
    for epoch in range(120):
        model.train(); tl=[]
        for xb,yb in dl(tr,True): xb,yb=xb.to(device),yb.to(device); opt.zero_grad(); z=model(xb); l=loss(z,yb); l.backward(); opt.step(); tl.append(l.item())
        model.eval(); vl=[]
        with torch.no_grad():
            for xb,yb in dl(va): vl.append(loss(model(xb.to(device)),yb.to(device)).item())
        v=float(np.mean(vl)); hist.append({'epoch':epoch+1,'train_loss':float(np.mean(tl)),'val_loss':v})
        if v<best-1e-5: best=v; patience=0; torch.save({'state_dict':model.state_dict(),'x_mean':torch.from_numpy(xm),'x_std':torch.from_numpy(xs),'y_mean':torch.from_numpy(ym),'y_std':torch.from_numpy(ys)},args.checkpoint/'best.pt')
        else: patience+=1
        if patience>=18: break
    ck=torch.load(args.checkpoint/'best.pt',map_location=device); model.load_state_dict(ck['state_dict']); model.eval(); outs=[]
    with torch.no_grad():
        for xb,_ in dl(te): outs.append(model(xb.to(device)).cpu().numpy())
    pred=np.concatenate(outs)*ys+ym; gt=y[te]; ridge=ridge_best[2].predict(xn[te]).reshape(-1,24,3)*ys+ym; mean=np.repeat(y[tr].mean(0)[None],len(te),0)
    mpraw=np.asarray([mp_skeleton(x[i])*1000 for i in te]); mproot=mpraw-mpraw[:,0:1]; gtroot=gt-gt[:,0:1]
    torso_mp=np.linalg.norm(mproot[:,12],axis=1).clip(1e-6); torso_gt=np.linalg.norm(gtroot[:,12],axis=1); mpscale=mproot*(torso_gt/torso_mp)[:,None,None]
    mppa=np.asarray([procrustes(p,g) for p,g in zip(mpraw,gt)])
    # Existing 2D corrector predictions for the same 3D subset/test rows.
    import importlib.util
    spec=importlib.util.spec_from_file_location('pilot',Path('training/train_human_pose_3d_pilot.py')); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); c2=torch.load(args.old_checkpoint/'best.pt',map_location='cpu'); m2=mod.Corrector(); m2.load_state_dict(c2['state_dict']); m2.eval(); b=np.asarray([mod.baseline(x[i][:132].reshape(33,4)) for i in te]);
    with torch.no_grad(): pred2=m2(torch.from_numpy(x[te,:132]),torch.from_numpy(b)).numpy()
    gt2=np.asarray([rows[i]['gt2'] for i in te]); two=pck2(pred2,gt2)
    methods={'train_mean':metric(mean,gt),'ridge':metric(ridge,gt),'mlp':metric(pred,gt),'mediapipe_raw_incompatible':metric(mpraw,gt),'mediapipe_root_aligned':metric(mproot,gtroot),'mediapipe_root_scale_aligned':metric(mpscale,gtroot),'mediapipe_procrustes_aligned':metric(mppa,gt)}
    # Per-view and posture metrics for selected MLP.
    byview={}
    for cam in sorted({rows[i]['camera'] for i in te}):
        q=[j for j,i in enumerate(te) if rows[i]['camera']==cam]; byview[str(cam)]={'count':len(q),**metric(pred[q],gt[q])}
    posture=defaultdict(list)
    for j,i in enumerate(te):
        g=gt[j]; shoulder=g[[16,17],1].mean(); wrist=g[[20,21],1].mean(); posture['arms_up' if wrist>shoulder else 'arms_down'].append(j)
        v1=g[16]-g[18]; v2=g[20]-g[18]; ang=math.degrees(math.acos(np.clip(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-9),-1,1))); posture['elbow_flexed' if ang<120 else 'elbow_extended'].append(j)
        dist=np.linalg.norm(g[[20,21]]-g[0],axis=1).mean(); posture['wrist_near_body' if dist<650 else 'wrist_far_body'].append(j)
    posture_metrics={k:{'count':len(q),**metric(pred[q],gt[q])} for k,q in posture.items() if q}
    # Per-sample/worst cases with conservative automatic reason tags.
    per=[]
    for j,i in enumerate(te):
        r=rows[i]; err=np.linalg.norm(pred[j,[LW,RW]]-gt[j,[LW,RW]],axis=1); axis=np.abs(pred[j,[LW,RW]]-gt[j,[LW,RW]]); lm=x[i][:132].reshape(33,4); g2=r['gt2']; reasons=[]
        if min(lm[15,3],lm[16,3])<.5: reasons.append('wrist_occlusion_or_low_visibility')
        if np.any((g2[[LW,RW]]<.03)|(g2[[LW,RW]]>.97)): reasons.append('wrist_near_or_outside_frame')
        if np.linalg.norm(g2[[LW,RW]].mean(0)-g2[0])<.18: reasons.append('arm_body_overlap')
        if np.linalg.norm(pred[j,LW]-gt[j,RW])<np.linalg.norm(pred[j,LW]-gt[j,LW]) or np.linalg.norm(pred[j,RW]-gt[j,LW])<np.linalg.norm(pred[j,RW]-gt[j,RW]): reasons.append('left_right_confusion')
        if axis[:,2].mean()>axis[:,:2].mean()*1.5: reasons.append('depth_ambiguity')
        img=cv2.imread(r['image'],cv2.IMREAD_GRAYSCALE); blur=float(cv2.Laplacian(img,cv2.CV_64F).var()) if img is not None else -1
        if 0<=blur<60: reasons.append('motion_blur_possible')
        if not reasons: reasons.append('other_or_domain_gap')
        per.append({'image':r['image'],'actor':r['actor'],'action':r['action'],'camera':r['camera'],'frame':r['frame'],'left_mpjpe':float(err[0]),'right_mpjpe':float(err[1]),'mean_wrist_mpjpe':float(err.mean()),'left_x_mae':float(axis[0,0]),'left_y_mae':float(axis[0,1]),'left_z_mae':float(axis[0,2]),'right_x_mae':float(axis[1,0]),'right_y_mae':float(axis[1,1]),'right_z_mae':float(axis[1,2]),'mediapipe_visibility_min':float(min(lm[15,3],lm[16,3])),'blur_score':blur,'reason':'|'.join(reasons)})
    per.sort(key=lambda r:r['mean_wrist_mpjpe'],reverse=True); fields=list(per[0]);
    for name,data in [('per_sample_errors.csv',per),('worst_cases.csv',per[:20])]:
        with (args.results/name).open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(data)
    with (args.results/'per_view_metrics.csv').open('w',newline='') as f:
        fields2=['camera','count','left_mpjpe','right_mpjpe','wrist_mean_mpjpe','left_x','left_y','left_z','right_x','right_y','right_z']; w=csv.DictWriter(f,fieldnames=fields2); w.writeheader()
        for cam,v in byview.items(): w.writerow({'camera':cam,'count':v['count'],'left_mpjpe':v['left_mpjpe'],'right_mpjpe':v['right_mpjpe'],'wrist_mean_mpjpe':v['wrist_mean_mpjpe'],'left_x':v['left_axis_mae']['x'],'left_y':v['left_axis_mae']['y'],'left_z':v['left_axis_mae']['z'],'right_x':v['right_axis_mae']['x'],'right_y':v['right_axis_mae']['y'],'right_z':v['right_axis_mae']['z']})
    reasons=Counter(z for r in per[:20] for z in r['reason'].split('|'))
    config={'seed':args.seed,'input':'MediaPipe 33 normalized landmarks + 33 world landmarks (full-body context)','output':'24 joints x 3 global AI-Hub coordinates','split_rule':'reuse original 2D actor+action groups','coordinate_interpretation':'global JSON units; values strongly consistent with millimetres, official unit not explicit','test_not_used_for_selection':True}
    report={'images':len(rows),'split_counts':{k:len(v) for k,v in split.items()},'detected_counts':{k:len(v) for k,v in ids.items()},'existing_2d_same_subset':two,'methods':methods,'per_view':byview,'posture':posture_metrics,'worst_reason_counts':reasons,'best_val_loss':best,'ridge_alpha':ridge_best[1]}
    (args.results/'metrics.json').write_text(json.dumps(report,indent=2)); (args.results/'experiment_config.json').write_text(json.dumps(config,indent=2)); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
