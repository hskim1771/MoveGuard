#!/usr/bin/env python3
"""Joint Human Pose + industrial masked pose refinement training."""
from __future__ import annotations
import csv,json,random,time
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from train_human_pose_3d_pilot import Corrector,WRISTS

SEED=20260804
def load_chunks(path):
 z=[np.load(p) for p in sorted(path.glob('chunk_*.npz'))]
 return {k:np.concatenate([x[k] for x in z]) for k in ('ids','split','x','base','gt','mask','norm_scale')}
def human_arrays(root,manifest):
 z=[np.load(p) for p in sorted((root/'data/human_pose_3d_sample/mediapipe_cache_50k_v1').glob('chunk_*.npz'))]
 a={k:np.concatenate([x[k] for x in z]) for k in ('ids','x','base','gt','detected')};ids=a['ids'].astype(str);pos={x:i for i,x in enumerate(ids)}
 def take(key):
  q=np.asarray([pos[x] for x in manifest['members'][key]]);q=q[a['detected'][q]];return q
 return a,take('train_50k'),take('validation_fixed'),take('test_fixed')
def wrist_pck(model,x,b,y,norm_scale=None):
 out=[];model.eval()
 with torch.no_grad():
  for i in range(0,len(x),512):out.append(model(torch.from_numpy(x[i:i+512]).to(next(model.parameters()).device),torch.from_numpy(b[i:i+512]).to(next(model.parameters()).device)).cpu().numpy())
 p=np.concatenate(out)
 if norm_scale is None:
  diag=np.linalg.norm(y.max(1)-y.min(1),axis=1).clip(1e-6);e=np.linalg.norm(p[:,WRISTS]-y[:,WRISTS],axis=2)/diag[:,None]
 else:e=np.linalg.norm((p[:,WRISTS]-y[:,WRISTS])*norm_scale[:,None,:],axis=2)
 return float((e<=.1).mean()),float((e<=.05).mean()),float(e.mean())
def main():
 root=Path(__file__).resolve().parents[1];out=root/'checkpoints/wrist_refiner_joint_domain_v1';res=root/'results/wrist_refiner_joint_domain_v1';out.mkdir(parents=True,exist_ok=True);res.mkdir(parents=True,exist_ok=True)
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
 manifest=json.loads((root/'results/wrist_refiner_large_scale_comparison/experiment_manifest.json').read_text());h,htr,hv,ht=human_arrays(root,manifest);ind=load_chunks(root/'data/human_pose_3d_sample/industrial_raw_cache_v1')
 itr=np.flatnonzero(ind['split'].astype(str)=='train');iv=np.flatnonzero(ind['split'].astype(str)=='validation')
 hx,hb,hy=h['x'][htr],h['base'][htr],h['gt'][htr];hm=np.ones((len(htr),24),bool)
 tx=np.concatenate([hx,ind['x'][itr]]);tb=np.concatenate([hb,ind['base'][itr]]);ty=np.concatenate([hy,ind['gt'][itr]]);tm=np.concatenate([hm,ind['mask'][itr]])
 domain=np.concatenate([np.zeros(len(htr),np.int64),np.ones(len(itr),np.int64)])
 device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');model=Corrector().to(device);opt=torch.optim.AdamW(model.parameters(),lr=7e-4,weight_decay=1e-4)
 ds=TensorDataset(torch.from_numpy(tx),torch.from_numpy(tb),torch.from_numpy(ty),torch.from_numpy(tm),torch.from_numpy(domain));dl=DataLoader(ds,batch_size=256,shuffle=True)
 best=-1;patience=0;hist=[];weights=torch.ones(24,device=device);weights[list(WRISTS)]=4
 for epoch in range(1,61):
  model.train();losses=[]
  for x,b,y,m,d in dl:
   x,b,y,m=x.to(device),b.to(device),y.to(device),m.to(device);opt.zero_grad();p=model(x,b);raw=nn.functional.smooth_l1_loss(p,y,beta=.02,reduction='none').mean(2);loss=(raw*m*weights).sum()/((m*weights).sum().clip(min=1));loss.backward();opt.step();losses.append(loss.item())
  hp10,hp05,hpe=wrist_pck(model,h['x'][hv],h['base'][hv],h['gt'][hv]);ip10,ip05,ipe=wrist_pck(model,ind['x'][iv],ind['base'][iv],ind['gt'][iv],ind['norm_scale'][iv]);score=.3*hp10+.7*ip10
  row={'epoch':epoch,'train_loss':float(np.mean(losses)),'human_val_pck_0_10':hp10,'industrial_val_pck_0_10':ip10,'selection_score':score};hist.append(row)
  with (res/'training_log.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=row);w.writeheader();w.writerows(hist)
  if score>best+1e-5:best=score;patience=0;torch.save({'state_dict':model.state_dict(),'seed':SEED,'selection':'0.3 human val + 0.7 industrial val wrist PCK@0.10'},out/'best.pt')
  else:patience+=1
  print(json.dumps({**row,'patience':patience}),flush=True)
  if patience>=12:break
 ck=torch.load(out/'best.pt',map_location=device);model.load_state_dict(ck['state_dict']);hp=wrist_pck(model,h['x'][ht],h['base'][ht],h['gt'][ht]);report={'seed':SEED,'architecture':'same Corrector 132 -> residual 24x2','loss':'masked SmoothL1 beta=.02, wrist weight 4','optimizer':'AdamW lr=7e-4','batch_size':256,'selection':'70% industrial validation + 30% Human Pose validation wrist PCK@0.10','counts':{'human_train':len(htr),'industrial_train':len(itr),'human_validation':len(hv),'industrial_validation':len(iv),'human_test':len(ht)},'best_selection_score':best,'human_test':{'pck_0_10':hp[0],'pck_0_05':hp[1],'mean_normalized_error':hp[2]},'epochs':hist,'checkpoint':'checkpoints/wrist_refiner_joint_domain_v1/best.pt'};(res/'training_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
