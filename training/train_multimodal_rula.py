#!/usr/bin/env python3
"""Train an image + MediaPipe feature multi-head posture model for RULA inputs."""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from backend.multimodal_rula import BINARY, CATEGORICAL, MultiModalRula
from backend.rula import calculate
from training.evaluate_rula_agreement import reference_components


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def device_for(name: str) -> torch.device:
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_records(manifest: Path, features: Path, limits: dict[str, int | None]) -> dict[str, list[dict]]:
    meta = {r["id"]: r for r in map(json.loads, manifest.read_text(encoding="utf-8").splitlines())}
    splits = {"train": [], "validation": [], "test": []}
    for line in features.open(encoding="utf-8"):
        feature = json.loads(line)
        split = feature.get("split")
        if feature.get("status") != "ok" or split not in splits or feature["id"] not in meta:
            continue
        if limits[split] is not None and len(splits[split]) >= limits[split]:
            continue
        splits[split].append({**meta[feature["id"]], "features": feature["features"]})
    return splits


class AihubDataset(Dataset):
    def __init__(self, records: list[dict], pose_mean: np.ndarray, pose_std: np.ndarray, size: int, augment: bool, cache: Path | None = None) -> None:
        self.records, self.mean, self.std, self.size, self.augment = records, pose_mean, pose_std, size, augment
        self.archives: dict[str, zipfile.ZipFile] = {}
        self.cached_images = self.cache_index = None
        if cache:
            info=json.loads((cache/"index.json").read_text(encoding="utf-8"));self.cache_index={value:i for i,value in enumerate(info["ids"])};self.cached_images=np.load(cache/"images.npy",mmap_mode="r")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        if self.cached_images is not None:
            pixels=np.asarray(self.cached_images[self.cache_index[record["id"]]],dtype=np.float32).copy()
            if self.augment and random.random()<.5:pixels=pixels[:,::-1].copy()
            pixels=pixels.transpose(2,0,1)/255.0
        else:
            archive_path = str(record["image_archive"])
            archive = self.archives.setdefault(archive_path, zipfile.ZipFile(archive_path))
            with Image.open(io.BytesIO(archive.read(record["image_member"]))) as image:
            # Decode the 4K JPEG at a reduced native scale before cropping. This
            # preserves the source ZIP and avoids resizing 8M pixels per epoch.
                original_width, original_height = image.size
                image.draft("RGB", (max(self.size * 4, 640), max(self.size * 4, 360)))
                image = image.convert("RGB")
                x1, y1, x2, y2 = map(float, record["bbox"])
                scale_x, scale_y = image.width / original_width, image.height / original_height
                x1, x2, y1, y2 = x1 * scale_x, x2 * scale_x, y1 * scale_y, y2 * scale_y
                px, py = (x2-x1)*.15, (y2-y1)*.15
                image = image.crop((max(0,x1-px), max(0,y1-py), min(image.width,x2+px), min(image.height,y2+py)))
                if self.augment and random.random() < .5:image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                image = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
                pixels = np.asarray(image, dtype=np.float32).transpose(2,0,1) / 255.0
        pose = (np.asarray(record["features"], dtype=np.float32) - self.mean) / self.std
        labels = record["labels"]
        cat = np.asarray([labels[k] for k in CATEGORICAL], dtype=np.int64)
        binary = np.asarray([labels.get(k, 0) for k in BINARY], dtype=np.float32)
        return torch.from_numpy(pixels), torch.from_numpy(pose), torch.from_numpy(cat), torch.from_numpy(binary)


def weights(records: list[dict], device: torch.device):
    cat=[]
    for key,n in CATEGORICAL.items():
        counts=Counter(r["labels"][key] for r in records); total=sum(counts.values())
        values=[total/(n*max(counts.get(i,0),1)) for i in range(n)]
        cat.append(torch.tensor(values,dtype=torch.float32,device=device))
    positives=np.asarray([[r["labels"].get(k,0) for k in BINARY] for r in records]).sum(axis=0)
    pos_weight=(len(records)-positives)/np.maximum(positives,1)
    return cat,torch.tensor(np.clip(pos_weight,1,20),dtype=torch.float32,device=device)


def run_epoch(model, loader, device, cat_loss, bin_loss, optimizer=None):
    training=optimizer is not None; model.train(training); total=0.; count=0
    for images,poses,cats,binary in loader:
        images,poses,cats,binary=images.to(device),poses.to(device),cats.to(device),binary.to(device)
        if training: optimizer.zero_grad(set_to_none=True)
        logits,binary_logits=model(images,poses)
        loss=sum(cat_loss[i](logits[k],cats[:,i]) for i,k in enumerate(CATEGORICAL))/len(CATEGORICAL)+bin_loss(binary_logits,binary)
        if training: loss.backward();optimizer.step()
        total+=float(loss.detach().cpu())*len(images);count+=len(images)
    return total/max(count,1)


@torch.inference_mode()
def evaluate(model,loader,device):
    model.eval(); truth={k:[] for k in CATEGORICAL}; pred={k:[] for k in CATEGORICAL}; bt=[];bp=[]
    for images,poses,cats,binary in loader:
        logits,binary_logits=model(images.to(device),poses.to(device))
        for i,k in enumerate(CATEGORICAL): truth[k]+=cats[:,i].tolist();pred[k]+=logits[k].argmax(1).cpu().tolist()
        bt.append(binary.numpy());bp.append((torch.sigmoid(binary_logits).cpu().numpy()>=.5).astype(int))
    metrics={k:{"accuracy":accuracy_score(truth[k],pred[k]),"macro_f1":f1_score(truth[k],pred[k],average="macro",zero_division=0),"weighted_f1":f1_score(truth[k],pred[k],average="weighted",zero_division=0)} for k in CATEGORICAL}
    bt=np.concatenate(bt);bp=np.concatenate(bp);metrics["adjustments"]={k:{"accuracy":accuracy_score(bt[:,i],bp[:,i]),"f1":f1_score(bt[:,i],bp[:,i],zero_division=0)} for i,k in enumerate(BINARY)}
    ref_scores=[];pred_scores=[];ref_actions=[];pred_actions=[]
    n=len(truth["neck"])
    for i in range(n):
        actual={k:int(truth[k][i]) for k in CATEGORICAL}; actual.update({k:int(bt[i,j]) for j,k in enumerate(BINARY)})
        guessed={k:int(pred[k][i]) for k in CATEGORICAL}; guessed.update({k:int(bp[i,j]) for j,k in enumerate(BINARY)})
        a=calculate({**reference_components(actual),"muscle_a":0,"force_a":0,"muscle_b":0,"force_b":0})
        b=calculate({**reference_components(guessed),"muscle_a":0,"force_a":0,"muscle_b":0,"force_b":0})
        ref_scores.append(a["final_score"]);pred_scores.append(b["final_score"]);ref_actions.append(a["action_level"]);pred_actions.append(b["action_level"])
    a,b=np.asarray(ref_scores),np.asarray(pred_scores)
    metrics["rula"]={"exact_agreement":float(np.mean(a==b)),"within_one_agreement":float(np.mean(np.abs(a-b)<=1)),"mean_absolute_error":float(np.mean(np.abs(a-b))),"action_level_agreement":float(np.mean(np.asarray(ref_actions)==np.asarray(pred_actions)))}
    return metrics


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument("--manifest",type=Path,required=True);p.add_argument("--features",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=12);p.add_argument("--batch-size",type=int,default=32);p.add_argument("--image-size",type=int,default=160);p.add_argument("--lr",type=float,default=1e-3);p.add_argument("--seed",type=int,default=20260803);p.add_argument("--device",default="auto")
    p.add_argument("--train-limit",type=int);p.add_argument("--validation-limit",type=int);p.add_argument("--test-limit",type=int);p.add_argument("--image-cache",type=Path);args=p.parse_args();seed_all(args.seed);device=device_for(args.device);args.output.mkdir(parents=True,exist_ok=True)
    records=load_records(args.manifest,args.features,{"train":args.train_limit,"validation":args.validation_limit,"test":args.test_limit})
    train_pose=np.asarray([r["features"] for r in records["train"]],dtype=np.float32);mean=train_pose.mean(0);std=np.maximum(train_pose.std(0),1e-5)
    datasets={s:AihubDataset(rows,mean,std,args.image_size,s=="train",args.image_cache) for s,rows in records.items()}
    loaders={s:DataLoader(ds,batch_size=args.batch_size,shuffle=s=="train",num_workers=0) for s,ds in datasets.items()}
    model=MultiModalRula(train_pose.shape[1]).to(device);cw,pw=weights(records["train"],device);cat_loss=[nn.CrossEntropyLoss(weight=w) for w in cw];bin_loss=nn.BCEWithLogitsLoss(pos_weight=pw);opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    best=-1.;history=[];started=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        train_loss=run_epoch(model,loaders["train"],device,cat_loss,bin_loss,opt);val=evaluate(model,loaders["validation"],device);score=val["rula"]["within_one_agreement"]
        history.append({"epoch":epoch,"train_loss":train_loss,"validation":val});print(json.dumps(history[-1],ensure_ascii=False),flush=True)
        if score>best:
            best=score;torch.save({"state_dict":model.state_dict(),"pose_mean":mean,"pose_std":std,"image_size":args.image_size,"categorical":CATEGORICAL,"binary":BINARY,"validation":val},args.output/"best.pt")
    checkpoint=torch.load(args.output/"best.pt",map_location=device,weights_only=False);model.load_state_dict(checkpoint["state_dict"]);test=evaluate(model,loaders["test"],device)
    report={"model":"multimodal_rula_multihead_v1","device":str(device),"seed":args.seed,"counts":{k:len(v) for k,v in records.items()},"seconds":time.perf_counter()-started,"best_validation_within_one":best,"history":history,"independent_test":test,"model_size_bytes":(args.output/"best.pt").stat().st_size}
    (args.output/"training_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
