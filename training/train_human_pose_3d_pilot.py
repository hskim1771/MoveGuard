#!/usr/bin/env python3
"""Pilot MediaPipe keypoint corrector trained on AI-Hub Human Pose 3D images."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

JOINTS = (
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
    "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar",
    "R_Collar", "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
)
DIRECT = (1, 2, 4, 5, 7, 8, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23)
WRISTS = (20, 21)


def baseline(lm: np.ndarray) -> np.ndarray:
    left_hand = lm[[17, 19, 21], :2].mean(0)
    right_hand = lm[[18, 20, 22], :2].mean(0)
    pelvis = lm[[23, 24], :2].mean(0)
    neck = lm[[11, 12], :2].mean(0)
    spine1 = pelvis * .75 + neck * .25
    spine2 = (pelvis + neck) / 2
    spine3 = pelvis * .25 + neck * .75
    return np.asarray([
        pelvis, lm[23,:2], lm[24,:2], spine1, lm[25,:2], lm[26,:2], spine2,
        lm[27,:2], lm[28,:2], spine3, lm[31,:2], lm[32,:2], neck,
        (neck+lm[11,:2])/2, (neck+lm[12,:2])/2, lm[0,:2], lm[11,:2], lm[12,:2],
        lm[13,:2], lm[14,:2], lm[15,:2], lm[16,:2], left_hand, right_hand,
    ], dtype=np.float32)


class Corrector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(33 * 4, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(.15),
            nn.Linear(256, 256), nn.GELU(), nn.Dropout(.10), nn.Linear(256, 24 * 2),
        )

    def forward(self, x, base):
        return base + self.net(x).reshape(-1, 24, 2) * .10


def split_groups(rows, seed):
    groups = sorted({(r['action'], r['actor']) for r in rows})
    random.Random(seed).shuffle(groups)
    n = len(groups); n_train = max(1, round(n*.70)); n_val = max(1, round(n*.15))
    sets = {'train': set(groups[:n_train]), 'val': set(groups[n_train:n_train+n_val]),
            'test': set(groups[n_train+n_val:])}
    return {k: [i for i,r in enumerate(rows) if (r['action'],r['actor']) in gs] for k,gs in sets.items()}, sets


def metrics(pred, gt, detected, indices):
    span = gt.max(1)-gt.min(1); diag=np.linalg.norm(span,axis=1).clip(1e-6)
    use = np.asarray(indices); pred=pred[:,use]; gt=gt[:,use]
    err=np.linalg.norm(pred-gt,axis=2); norm=err/diag[:,None]
    pixel_err=np.linalg.norm((pred-gt)*np.asarray([1920.,1080.],dtype=np.float32),axis=2)
    return {'images':int(len(gt)), 'detection_rate':float(detected.mean()),
            'mean_pixel_error':float(pixel_err.mean()),
            'mean_normalized_error':float(norm.mean()),
            'pck_0_05':float((norm<=.05).mean()), 'pck_0_10':float((norm<=.10).mean())}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--model',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=20260804); ap.add_argument('--epochs',type=int,default=80)
    args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    manifest=json.loads((args.base/'labels_selected/selection_manifest.json').read_text())
    rows=[]
    for r in manifest:
        d=json.loads((args.base/r['label_2d']).read_text()); info=d['info']; ann=d['annotations']
        gt=(np.asarray(ann['2d_pos'],np.float32)/np.asarray([info['img_width'],info['img_height']],np.float32)).astype(np.float32)
        rows.append({'image':args.base/r['image'], 'gt':gt,
                     'action':str(info['action_category_id']), 'actor':info['actor_id'], 'camera':int(info['camera_no']), 'frame':int(ann['frame_no'])})
    splits, group_sets=split_groups(rows,args.seed)
    (args.output/'split_manifest.json').write_text(json.dumps({'seed':args.seed,'groups':{k:[list(x) for x in sorted(v)] for k,v in group_sets.items()},'counts':{k:len(v) for k,v in splits.items()}},indent=2))
    cache=args.output/'mediapipe_cache.npz'
    if cache.exists():
        c=np.load(cache); x,base,gt,detected=c['x'].astype(np.float32),c['base'].astype(np.float32),c['gt'].astype(np.float32),c['detected']
    else:
        opts=vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(args.model)),running_mode=vision.RunningMode.IMAGE,num_poses=1,min_pose_detection_confidence=.3,min_pose_presence_confidence=.3)
        detector=vision.PoseLandmarker.create_from_options(opts); xs=[]; bases=[]; gts=[]; ds=[]; started=time.time()
        for n,r in enumerate(rows,1):
            bgr=cv2.imread(str(r['image'])); rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            result=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb))
            ok=bool(result.pose_landmarks); ds.append(ok); gts.append(r['gt'])
            if ok:
                lm=np.asarray([[p.x,p.y,p.z,p.visibility] for p in result.pose_landmarks[0]],np.float32)
            else: lm=np.zeros((33,4),np.float32)
            xs.append(lm.reshape(-1)); bases.append(baseline(lm))
            if n%250==0: print(json.dumps({'processed':n,'total':len(rows),'seconds':round(time.time()-started,1)}),flush=True)
        detector.close(); x=np.asarray(xs); base=np.asarray(bases); gt=np.asarray(gts); detected=np.asarray(ds)
        np.savez_compressed(cache,x=x,base=base,gt=gt,detected=detected)
    # Only detected samples train; failed detections remain part of reported detection rate.
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu'); model=Corrector().to(device)
    train=[i for i in splits['train'] if detected[i]]; val=[i for i in splits['val'] if detected[i]]; test=[i for i in splits['test'] if detected[i]]
    def loader(idx,shuffle=False):
        ds=TensorDataset(torch.from_numpy(x[idx]),torch.from_numpy(base[idx]),torch.from_numpy(gt[idx]))
        return DataLoader(ds,batch_size=128,shuffle=shuffle)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); loss_fn=nn.SmoothL1Loss(beta=.02)
    best=float('inf'); patience=0; history=[]
    for epoch in range(args.epochs):
        model.train(); losses=[]
        for xb,bb,yb in loader(train,True):
            xb,bb,yb=xb.to(device),bb.to(device),yb.to(device); opt.zero_grad(); pred=model(xb,bb)
            loss=loss_fn(pred,yb); loss.backward(); opt.step(); losses.append(loss.item())
        model.eval(); vals=[]
        with torch.no_grad():
            for xb,bb,yb in loader(val): vals.append(loss_fn(model(xb.to(device),bb.to(device)),yb.to(device)).item())
        v=float(np.mean(vals)); history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'val_loss':v})
        if v<best-1e-6:
            best=v; patience=0; torch.save({'state_dict':model.state_dict(),'joints':JOINTS,'seed':args.seed},args.output/'best.pt')
        else: patience+=1
        if epoch%5==0: print(history[-1],flush=True)
        if patience>=12: break
    model.load_state_dict(torch.load(args.output/'best.pt',map_location=device)['state_dict']); model.eval()
    def predict(idx):
        out=[]
        with torch.no_grad():
            for xb,bb,_ in loader(idx): out.append(model(xb.to(device),bb.to(device)).cpu().numpy())
        return np.concatenate(out)
    pred=predict(test); test_gt=gt[test]; test_base=base[test]; test_det=detected[splits['test']]
    report={'seed':args.seed,'device':str(device),'split_counts':{k:len(v) for k,v in splits.items()},'detected_counts':{'train':len(train),'val':len(val),'test':len(test)},
            'baseline':{'all_24':metrics(test_base,test_gt,test_det,range(24)),'direct_16':metrics(test_base,test_gt,test_det,DIRECT),'wrists':metrics(test_base,test_gt,test_det,WRISTS)},
            'corrected':{'all_24':metrics(pred,test_gt,test_det,range(24)),'direct_16':metrics(pred,test_gt,test_det,DIRECT),'wrists':metrics(pred,test_gt,test_det,WRISTS)},'best_val_loss':best,'epochs':history}
    (args.output/'training_report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))


if __name__=='__main__': main()
