#!/usr/bin/env python3
"""Evaluate industrial_clean_aug_v1 on the fixed independent test cache."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.evaluate_wrist_temporal_pipeline import GT_WRIST, JOINT_WRIST, frame_rula, human_confidence, wrist
from training.train_human_pose_3d_pilot import Corrector, baseline

COMMON = {"pelvis_center":(0,18),"left_hip":(1,19),"right_hip":(2,23),"left_knee":(4,20),"right_knee":(5,24),
          "spine":(6,17),"left_ankle":(7,21),"right_ankle":(8,25),"left_foot":(10,22),"right_foot":(11,26),
          "neck":(12,4),"left_shoulder":(16,5),"right_shoulder":(17,11),"left_elbow":(18,6),"right_elbow":(19,12),
          "left_wrist":(20,7),"right_wrist":(21,13)}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--cache",type=Path,default=ROOT/"experiments/wrist_rula_pipeline_validation/independent_test_pose_cache.jsonl")
    parser.add_argument("--checkpoint",type=Path,default=ROOT/"checkpoints/industrial_clean_aug_v1/best.pt")
    parser.add_argument("--output",type=Path,default=ROOT/"results/industrial_clean_aug_v1/independent_test_report.json")
    args=parser.parse_args()
    records=[json.loads(line) for line in args.cache.read_text().splitlines() if json.loads(line).get("status")=="ok"]
    features=[];bases=[]
    for row in records:
        raw=np.asarray(row["raw"],np.float32);conf=np.asarray(row["raw_confidence"],np.float32)
        x=np.column_stack([raw,conf]).reshape(-1);features.append(x);bases.append(baseline(x.reshape(33,4)))
    features=np.asarray(features,np.float32);bases=np.asarray(bases,np.float32)
    model=Corrector();model.load_state_dict(torch.load(args.checkpoint,map_location="cpu",weights_only=False)["state_dict"]);model.eval()
    pred=[]
    with torch.inference_mode():
        for start in range(0,len(features),512):pred.append(model(torch.from_numpy(features[start:start+512]),torch.from_numpy(bases[start:start+512])).numpy())
    pred=np.concatenate(pred)
    errors=defaultdict(list);angle_errors=[];score_match=[];rula_match=[]
    for index,row in enumerate(records):
        gt=np.asarray(row["gt"],np.float32);gt_conf=gt[:,2]/2;scale=np.asarray(row["norm_scale"],np.float32)
        conf=human_confidence(np.asarray(row["raw_confidence"],np.float32))
        for name,(human,aihub) in COMMON.items():errors[name].append(float(np.linalg.norm((pred[index,human]-gt[aihub,:2])*scale)))
        for side in ("left","right"):
            pm=wrist(pred[index],conf,JOINT_WRIST,side);gm=wrist(gt,gt_conf,GT_WRIST,side)
            if pm.valid and gm.valid:
                angle_errors.append(abs(float(pm.angle)-float(gm.angle)));score_match.append(int(pm.rula_score==gm.rula_score))
        pr=frame_rula(pred[index],conf,"joint");gr=frame_rula(gt,gt_conf,"gt")
        if pr is not None and gr is not None:rula_match.append(int(pr==gr))
    per_joint={name:{"pck_0_10":float((np.asarray(values)<=.1).mean()),"pck_0_05":float((np.asarray(values)<=.05).mean()),"count":len(values)} for name,values in errors.items()}
    all_errors=np.concatenate([np.asarray(values) for values in errors.values()])
    report={"model":"industrial_clean_aug_v1","independent_test":True,"evaluated_detected_images":len(records),
            "overall_common_pck_0_10":float((all_errors<=.1).mean()),"overall_common_pck_0_05":float((all_errors<=.05).mean()),
            "per_joint":per_joint,"wrist_angle_mae":float(np.mean(angle_errors)),"wrist_angle_within_10":float((np.asarray(angle_errors)<=10).mean()),
            "wrist_score_agreement":float(np.mean(score_match)),"wrist_score_count":len(score_match),
            "derived_frame_rula_agreement":float(np.mean(rula_match)),"derived_frame_rula_count":len(rula_match),
            "notes":["GT keypoint-derived 2D angles/scores; not expert final RULA accuracy","test cache was not used for model selection"]}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
