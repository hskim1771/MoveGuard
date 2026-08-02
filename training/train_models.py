#!/usr/bin/env python3
"""Compare linear, gradient boosting, and PyTorch MLP without test-set tuning."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import joblib, numpy as np
import torch
from torch import nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

TARGETS = ("neck", "trunk", "upperarm", "lowerarm", "legs")

def load(path: Path):
    rows=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
    rows=[r for r in rows if r.get("status")=="ok" and all(r.get("labels",{}).get(k) is not None for k in TARGETS)]
    x=np.asarray([r["features"] for r in rows],dtype=np.float32)
    y=np.asarray(["|".join(str(r["labels"][k]) for k in TARGETS) for r in rows])
    s=np.asarray([r["split"] for r in rows])
    return x,y,s

def metrics(model,x,y):
    t=time.perf_counter(); pred=model.predict(x); elapsed=time.perf_counter()-t
    return {"accuracy":accuracy_score(y,pred),"macro_f1":f1_score(y,pred,average="macro",zero_division=0),"weighted_f1":f1_score(y,pred,average="weighted",zero_division=0),"classification_report":classification_report(y,pred,output_dict=True,zero_division=0),"confusion_matrix":confusion_matrix(y,pred).tolist(),"inference_ms_per_sample":elapsed/max(1,len(y))*1000}

class TorchPredictor:
    def __init__(self, net, encoder, mean, std): self.net,self.encoder,self.mean,self.std=net,encoder,mean,std
    def predict(self,x):
        self.net.eval()
        with torch.no_grad(): idx=self.net(torch.tensor((x-self.mean)/self.std,dtype=torch.float32)).argmax(1).numpy()
        return self.encoder.inverse_transform(idx)

def train_mlp(x,y,train_mask,seed):
    torch.manual_seed(seed); enc=LabelEncoder().fit(y[train_mask]); yi=enc.transform(y[train_mask])
    mean=x[train_mask].mean(0); std=x[train_mask].std(0)+1e-6
    net=nn.Sequential(nn.Linear(x.shape[1],256),nn.ReLU(),nn.Dropout(.2),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,len(enc.classes_)))
    opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4); loss_fn=nn.CrossEntropyLoss()
    tx=torch.tensor((x[train_mask]-mean)/std,dtype=torch.float32); ty=torch.tensor(yi,dtype=torch.long)
    net.train()
    for _ in range(30):
        for start in range(0,len(tx),512):
            opt.zero_grad(); loss=loss_fn(net(tx[start:start+512]),ty[start:start+512]); loss.backward(); opt.step()
    return TorchPredictor(net,enc,mean,std)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--features",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--seed",type=int,default=20260803); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    x,y,s=load(a.features); tr=s=="train"; va=s=="validation"; te=s=="test"
    candidates={"linear":make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight="balanced",random_state=a.seed)),"gradient_boosting":HistGradientBoostingClassifier(random_state=a.seed)}
    validation={}
    for name,model in candidates.items():
        model.fit(x[tr],y[tr]); validation[name]=metrics(model,x[va],y[va]); joblib.dump(model,a.output/f"{name}.joblib")
    mlp=train_mlp(x,y,tr,a.seed); candidates["pytorch_mlp"]=mlp; validation["pytorch_mlp"]=metrics(mlp,x[va],y[va]); torch.save({"state_dict":mlp.net.state_dict(),"classes":mlp.encoder.classes_.tolist(),"mean":mlp.mean,"std":mlp.std},a.output/"pytorch_mlp.pt")
    winner=max(validation,key=lambda n:validation[n]["macro_f1"]); final=candidates[winner]
    # Test is touched exactly once after validation-only selection.
    test=metrics(final,x[te],y[te]); model_path=a.output/("final.pt" if winner=="pytorch_mlp" else "final.joblib")
    if winner=="pytorch_mlp": torch.save({"state_dict":final.net.state_dict(),"classes":final.encoder.classes_.tolist(),"mean":final.mean,"std":final.std},model_path)
    else: joblib.dump(final,model_path)
    report={"seed":a.seed,"target":"composite posture (neck|trunk|upperarm|lowerarm|legs)","counts":{"train":int(tr.sum()),"validation":int(va.sum()),"test":int(te.sum())},"validation":validation,"selected_model":winner,"test":test,"model_size_bytes":model_path.stat().st_size,"rula_validation":"평가 불가"}
    (a.output/"training_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
