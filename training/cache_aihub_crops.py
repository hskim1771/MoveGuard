#!/usr/bin/env python3
"""Create a resumable uint8 crop cache without modifying AI-Hub source ZIPs."""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

_ARCHIVES: dict[str, zipfile.ZipFile] = {}


def render_crop(task: tuple[int, dict, int]) -> tuple[int, str, np.ndarray]:
    i, r, size = task
    path=str(r["image_archive"]);zf=_ARCHIVES.setdefault(path,zipfile.ZipFile(path))
    with Image.open(io.BytesIO(zf.read(r["image_member"]))) as im:
        ow,oh=im.size;im.draft("RGB",(max(size*4,640),max(size*4,360)));im=im.convert("RGB")
        x1,y1,x2,y2=map(float,r["bbox"]);sx,sy=im.width/ow,im.height/oh;x1,x2,y1,y2=x1*sx,x2*sx,y1*sy,y2*sy;px,py=(x2-x1)*.15,(y2-y1)*.15
        im=im.crop((max(0,x1-px),max(0,y1-py),min(im.width,x2+px),min(im.height,y2+py))).resize((size,size),Image.Resampling.BILINEAR)
        return i, r["id"], np.asarray(im,dtype=np.uint8)


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument("--manifest",type=Path,required=True);p.add_argument("--features",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--size",type=int,default=128);p.add_argument("--workers",type=int,default=6);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    meta={r["id"]:r for r in map(json.loads,args.manifest.read_text(encoding="utf-8").splitlines())}
    records=[]
    for line in args.features.open(encoding="utf-8"):
        r=json.loads(line)
        if r.get("status")=="ok" and r["id"] in meta:records.append(meta[r["id"]])
    index_path=args.output/"index.json";image_path=args.output/"images.npy";checkpoint_path=args.output/"checkpoint.json"
    if image_path.exists():images=np.lib.format.open_memmap(image_path,mode="r+")
    else:images=np.lib.format.open_memmap(image_path,mode="w+",dtype=np.uint8,shape=(len(records),args.size,args.size,3))
    start=0
    if checkpoint_path.exists():start=int(json.loads(checkpoint_path.read_text()).get("processed",0))
    tasks=((i,records[i],args.size) for i in range(start,len(records)))
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i,record_id,pixels in pool.imap(render_crop,tasks,chunksize=8):
            images[i]=pixels
            if (i+1)%100==0 or i+1==len(records):
                images.flush();checkpoint_path.write_text(json.dumps({"processed":i+1,"total":len(records),"last_id":record_id},ensure_ascii=False,indent=2)+"\n")
                if (i+1)%1000==0:print(json.dumps({"processed":i+1,"total":len(records)}),flush=True)
    index_path.write_text(json.dumps({"size":args.size,"ids":[r["id"] for r in records]},ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"complete":True,"count":len(records),"images":str(image_path),"index":str(index_path)}),flush=True)

if __name__=="__main__":main()
