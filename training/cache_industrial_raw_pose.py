#!/usr/bin/env python3
"""Cache raw crop-normalized MediaPipe/AI-Hub joints for joint-domain training."""
from __future__ import annotations
import argparse,io,json,os,time,zipfile
from pathlib import Path
import mediapipe as mp,numpy as np
from PIL import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from train_human_pose_3d_pilot import baseline

STRICT={0:18,1:19,2:23,6:17,4:20,5:24,7:21,8:25,10:22,11:26,12:4,16:5,17:11,18:6,19:12,20:7,21:13}
DETECTOR=None;ARCHIVES={}

def init_worker(model):
 global DETECTOR
 DETECTOR=vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=model),running_mode=vision.RunningMode.IMAGE,num_poses=1,min_pose_detection_confidence=.3,min_pose_presence_confidence=.3))

def process(r):
 try:
  iz=ARCHIVES.setdefault(r['image_archive'],zipfile.ZipFile(r['image_archive']));lz=ARCHIVES.setdefault(r['label_archive'],zipfile.ZipFile(r['label_archive']))
  label=json.loads(lz.read(r['label_member']));kp=np.asarray(label['annotations'][0]['keypoints'],np.float32)
  with Image.open(io.BytesIO(iz.read(r['image_member']))) as im:
   original_width,original_height=im.size;im.draft('RGB',(960,540));im=im.convert('RGB')
   sx,sy=im.width/original_width,im.height/original_height;kp[:,:2]*=np.asarray([sx,sy],np.float32)
   x1,y1,x2,y2=map(float,r['bbox']);x1*=sx;x2*=sx;y1*=sy;y2*=sy;px,py=(x2-x1)*.15,(y2-y1)*.15
   left,top=max(0.,x1-px),max(0.,y1-py);right,bottom=min(im.width,x2+px),min(im.height,y2+py)
   rgb=np.asarray(im.crop((left,top,right,bottom)))
  z=DETECTOR.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb))
  if not z.pose_landmarks:return {'id':r['id'],'status':'no_person','split':r['split']}
  lm=np.asarray([[p.x,p.y,p.z,p.visibility] for p in z.pose_landmarks[0]],np.float32);base=baseline(lm)
  gt=(kp[:,:2]-np.asarray([left,top],np.float32))/np.asarray([right-left,bottom-top],np.float32)
  target=base.copy();mask=np.zeros(24,bool)
  for hp,ai in STRICT.items():target[hp]=gt[ai];mask[hp]=True
  norm_scale=np.asarray([right-left,bottom-top],np.float32)/max(float(np.hypot(x2-x1,y2-y1)),1.)
  return {'id':r['id'],'status':'ok','split':r['split'],'x':lm.reshape(-1),'base':base,'gt':target,'mask':mask,'norm_scale':norm_scale}
 except Exception as e:return {'id':r.get('id'),'status':'error','split':r.get('split'),'reason':f'{type(e).__name__}: {e}'}

def main():
 root=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,default=root/'artifacts/aihub/audit/split_manifest.jsonl');p.add_argument('--features',type=Path,default=root/'artifacts/aihub/full/features.jsonl');p.add_argument('--output',type=Path,default=root/'data/human_pose_3d_sample/industrial_raw_cache_v1');p.add_argument('--model',type=Path,default=root/'mediapipe/models/pose_landmarker_lite.task');p.add_argument('--workers',type=int,default=8);p.add_argument('--chunk-size',type=int,default=250);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 eligible={z['id'] for z in map(json.loads,a.features.read_text().splitlines()) if z.get('status')=='ok' and z.get('split') in ('train','validation')}
 records=[z for z in map(json.loads,a.manifest.read_text().splitlines()) if z['id'] in eligible and z['split'] in ('train','validation')]
 done=set()
 for f in a.output.glob('chunk_*.npz'):done.update(np.load(f)['ids'].astype(str))
 pending=[r for r in records if r['id'] not in done];started=time.time();buffer=[];fail=a.output/'failures.jsonl'
 from multiprocessing import get_context
 with get_context('spawn').Pool(a.workers,init_worker,(str(a.model),)) as pool,fail.open('a') as log:
  for n,row in enumerate(pool.imap_unordered(process,pending,chunksize=8),1):
   if row['status']=='ok':buffer.append(row)
   else:log.write(json.dumps(row,ensure_ascii=False)+'\n')
   if len(buffer)>=a.chunk_size:
    first=len(list(a.output.glob('chunk_*.npz')));np.savez_compressed(a.output/f'chunk_{first:04d}.npz',ids=np.asarray([x['id'] for x in buffer]),split=np.asarray([x['split'] for x in buffer]),x=np.asarray([x['x'] for x in buffer],np.float32),base=np.asarray([x['base'] for x in buffer],np.float32),gt=np.asarray([x['gt'] for x in buffer],np.float32),mask=np.asarray([x['mask'] for x in buffer],bool),norm_scale=np.asarray([x['norm_scale'] for x in buffer],np.float32));buffer=[]
   if n%100==0:
    completed=len(done)+n;elapsed=time.time()-started;print(json.dumps({'phase':'industrial_raw_cache','completed':completed,'total':len(records),'percent':round(completed/len(records)*100,2),'this_run_rate':round(n/elapsed,2),'eta_minutes':round((len(pending)-n)/(n/elapsed)/60,1)}),flush=True)
 if buffer:
  first=len(list(a.output.glob('chunk_*.npz')));np.savez_compressed(a.output/f'chunk_{first:04d}.npz',ids=np.asarray([x['id'] for x in buffer]),split=np.asarray([x['split'] for x in buffer]),x=np.asarray([x['x'] for x in buffer],np.float32),base=np.asarray([x['base'] for x in buffer],np.float32),gt=np.asarray([x['gt'] for x in buffer],np.float32),mask=np.asarray([x['mask'] for x in buffer],bool),norm_scale=np.asarray([x['norm_scale'] for x in buffer],np.float32))
 print(json.dumps({'phase':'industrial_raw_cache_complete','requested':len(records),'chunks':len(list(a.output.glob('chunk_*.npz')))}),flush=True)

if __name__=='__main__':main()
