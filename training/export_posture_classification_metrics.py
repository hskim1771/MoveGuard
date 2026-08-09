#!/usr/bin/env python3
"""Reproduce detailed independent-test metrics for the existing posture model."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from train_multimodal_rula import AihubDataset, load_records
from backend.multimodal_rula import CATEGORICAL, MultiModalRula


def main():
    root=Path(__file__).resolve().parents[1]; p=argparse.ArgumentParser()
    p.add_argument("--manifest",type=Path,default=root/"artifacts/aihub/audit/split_manifest.jsonl")
    p.add_argument("--features",type=Path,default=root/"artifacts/aihub/full/features.jsonl")
    p.add_argument("--image-cache",type=Path,default=root/"artifacts/aihub/image_cache_128")
    p.add_argument("--checkpoint",type=Path,default=root/"artifacts/aihub/multimodal_full/best.pt")
    p.add_argument("--output",type=Path,default=root/"results/wrist_refiner_large_scale_comparison")
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    records=load_records(args.manifest,args.features,{"train":None,"validation":None,"test":None})
    ck=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    ds=AihubDataset(records["test"],ck["pose_mean"],ck["pose_std"],ck["image_size"],False,args.image_cache)
    loader=DataLoader(ds,batch_size=128,shuffle=False,num_workers=0)
    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model=MultiModalRula(len(ck["pose_mean"])).to(device);model.load_state_dict(ck["state_dict"]);model.eval()
    truth={k:[] for k in CATEGORICAL};pred={k:[] for k in CATEGORICAL}
    with torch.no_grad():
        for images,poses,cats,_ in loader:
            logits,_=model(images.to(device),poses.to(device))
            for i,key in enumerate(CATEGORICAL):
                truth[key].extend(cats[:,i].tolist());pred[key].extend(logits[key].argmax(1).tolist())
    summary=[];classes=[];confusions={}
    for key,nclasses in CATEGORICAL.items():
        y,z=np.asarray(truth[key]),np.asarray(pred[key]);labels=list(range(nclasses))
        pr,rc,f1,sup=precision_recall_fscore_support(y,z,labels=labels,zero_division=0)
        macro=precision_recall_fscore_support(y,z,average="macro",zero_division=0)
        weighted=precision_recall_fscore_support(y,z,average="weighted",zero_division=0)
        summary.append({"target":key,"accuracy":accuracy_score(y,z),"macro_precision":macro[0],"macro_recall":macro[1],
                        "macro_f1":macro[2],"weighted_f1":weighted[2],"support":len(y),
                        "final_pipeline_note":"unchanged: wrist refiner is downstream of this frozen classifier"})
        classes.extend({"target":key,"class":i,"precision":pr[i],"recall":rc[i],"f1":f1[i],"support":sup[i]} for i in labels)
        confusions[key]=confusion_matrix(y,z,labels=labels).tolist()
    for name,rows in (("posture_classification_metrics.csv",summary),("posture_class_metrics.csv",classes)):
        with (args.output/name).open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    (args.output/"posture_confusion_matrices.json").write_text(json.dumps(confusions,indent=2))
    provenance={"existing_71_7_trace":"0.717103 is validation binary F1 for neck_bending, not an overall posture accuracy",
                "checkpoint":str(args.checkpoint),"test_count":len(ds),"metrics":summary}
    (args.output/"posture_classification_report.json").write_text(json.dumps(provenance,indent=2))
    print(json.dumps(provenance,indent=2))


if __name__=="__main__":main()
