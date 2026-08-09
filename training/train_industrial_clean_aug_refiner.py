#!/usr/bin/env python3
"""Train an industrial-only refiner on the clean photometric augmentation cache."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_human_pose_3d_pilot import Corrector

ROOT = Path(__file__).resolve().parents[1]
WRISTS = (20, 21); HANDS = (22, 23); ELBOWS = (18, 19)


def load(path: Path, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    chunks = [np.load(file) for file in sorted(path.glob("chunk_*.npz"))]
    return {key: np.concatenate([chunk[key] for chunk in chunks]) for key in keys}


def angle_vector(points: torch.Tensor, side: int) -> torch.Tensor:
    elbow, wrist, hand = (18, 20, 22) if side == 0 else (19, 21, 23)
    forearm = torch.nn.functional.normalize(points[:, wrist] - points[:, elbow], dim=1)
    hand_vector = torch.nn.functional.normalize(points[:, hand] - points[:, wrist], dim=1)
    dot = (forearm * hand_vector).sum(1).clamp(-1, 1)
    cross = forearm[:, 0] * hand_vector[:, 1] - forearm[:, 1] * hand_vector[:, 0]
    return torch.atan2(cross, dot)


def metrics(model: nn.Module, arrays: dict[str, np.ndarray], indices: np.ndarray, device: torch.device) -> dict[str, float]:
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), 512):
            batch = indices[start:start+512]
            predictions.append(model(torch.from_numpy(arrays["x"][batch]).to(device), torch.from_numpy(arrays["base"][batch]).to(device)).cpu().numpy())
    pred = np.concatenate(predictions); gt = arrays["gt"][indices]; scale = arrays["norm_scale"][indices]
    error = np.linalg.norm((pred - gt) * scale[:, None, :], axis=2)
    common = arrays["mask"][indices]
    wrist_error = error[:, list(WRISTS)]
    return {"overall_pck_0_10": float((error[common] <= .1).mean()), "wrist_pck_0_10": float((wrist_error <= .1).mean()),
            "wrist_pck_0_05": float((wrist_error <= .05).mean())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, default=ROOT / "data/human_pose_3d_sample/industrial_clean_aug_cache_v1")
    parser.add_argument("--validation-cache", type=Path, default=ROOT / "data/human_pose_3d_sample/industrial_raw_cache_v1")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints/industrial_clean_aug_v1")
    parser.add_argument("--result-dir", type=Path, default=ROOT / "results/industrial_clean_aug_v1")
    parser.add_argument("--epochs", type=int, default=50); parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args(); args.checkpoint_dir.mkdir(parents=True, exist_ok=True); args.result_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    train = load(args.train_cache, ("x","base","gt","mask","norm_scale"))
    validation = load(args.validation_cache, ("split","x","base","gt","mask","norm_scale"))
    val_indices = np.flatnonzero(validation["split"].astype(str) == "validation")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = Corrector().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    dataset = TensorDataset(*(torch.from_numpy(train[key]) for key in ("x","base","gt","mask")))
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    weights = torch.ones(24, device=device); weights[list(ELBOWS)] = 2; weights[list(HANDS)] = 3; weights[list(WRISTS)] = 4
    history=[]; best=-1.; patience=0
    for epoch in range(1, args.epochs+1):
        model.train(); losses=[]
        for x, base, gt, mask in loader:
            x,base,gt,mask=x.to(device),base.to(device),gt.to(device),mask.to(device)
            optimizer.zero_grad(); pred=model(x,base)
            coordinate=torch.nn.functional.smooth_l1_loss(pred,gt,beta=.02,reduction="none").mean(2)
            coordinate=(coordinate*mask*weights).sum()/(mask*weights).sum().clamp(min=1)
            angle_loss=0.0
            for side in (0,1):
                difference=angle_vector(pred,side)-angle_vector(gt,side)
                wrapped=torch.atan2(torch.sin(difference),torch.cos(difference))
                angle_loss=angle_loss+torch.nn.functional.smooth_l1_loss(wrapped,torch.zeros_like(wrapped),beta=.10)
            angle_loss=angle_loss/2
            loss=coordinate + .15*angle_loss; loss.backward(); optimizer.step(); losses.append(float(loss.item()))
        result=metrics(model,validation,val_indices,device)
        # The legacy validation cache has no hand-center GT; angle is evaluated
        # only after selection on the untouched independent test cache.
        selection=.50*result["overall_pck_0_10"]+.50*result["wrist_pck_0_10"]
        row={"epoch":epoch,"train_loss":float(np.mean(losses)),**result,"selection_score":selection};history.append(row)
        with (args.result_dir/"training_log.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=row);writer.writeheader();writer.writerows(history)
        if selection>best+1e-5:
            best=selection;patience=0;torch.save({"state_dict":model.state_dict(),"seed":args.seed,"selection":"industrial validation composite"},args.checkpoint_dir/"best.pt")
        else: patience+=1
        print(json.dumps({**row,"patience":patience}),flush=True)
        if patience>=10: break
    report={"model":"industrial_clean_aug_v1","source":"clean industrial train only","train_samples":len(train["x"]),"validation_samples":len(val_indices),
            "seed":args.seed,"best_selection_score":best,"history":history,"checkpoint":str(args.checkpoint_dir/"best.pt")}
    (args.result_dir/"training_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")


if __name__ == "__main__": main()
