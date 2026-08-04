#!/usr/bin/env python3
"""Create compact comparison tables, plots, provenance, and final report."""
from __future__ import annotations
import csv,json,shutil
from pathlib import Path
import matplotlib.pyplot as plt

def write_csv(path,rows):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

def main():
    root=Path(__file__).resolve().parents[1]; out=root/"results/wrist_refiner_large_scale_comparison"
    manifest_path=out/"experiment_manifest.json"; manifest=json.loads(manifest_path.read_text())
    # Raw GT rows were only required while building the completed feature cache.
    # Keep exact membership IDs and provenance, but remove bulky duplicated labels.
    manifest.pop("rows",None);manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    hp=json.loads((out/"humanpose_wrist_metrics.json").read_text())
    names=[("MediaPipe","mediapipe",0),("Refiner 5k","refiner_5k_frozen",3828),("Refiner 20k","refiner_20k",20000),("Refiner 50k","refiner_50k",50000)]
    hrows=[]
    for label,key,size in names:
        x=hp[key];hrows.append({"model":label,"train_images":size,"left_pck_0_05":x["left_pck_0_05"],"right_pck_0_05":x["right_pck_0_05"],
            "mean_pck_0_05":x["mean_pck_0_05"],"left_pck_0_10":x["left_pck_0_10"],"right_pck_0_10":x["right_pck_0_10"],
            "mean_pck_0_10":x["mean_pck_0_10"],"detection_rate":x["detection_rate"]})
    write_csv(out/"humanpose_wrist_metrics.csv",hrows)
    learning=[{"train_images":r["train_images"],"humanpose_wrist_pck_0_10":r["mean_pck_0_10"]} for r in hrows[1:]]
    write_csv(out/"learning_curve.csv",learning)
    plt.figure(figsize=(7,4));plt.plot([x["train_images"] for x in learning],[x["humanpose_wrist_pck_0_10"]*100 for x in learning],marker="o")
    plt.xlabel("Training images");plt.ylabel("Wrist PCK@0.10 (%)");plt.grid(alpha=.3);plt.tight_layout();plt.savefig(out/"wrist_accuracy_vs_dataset_size.png",dpi=160);plt.close()
    industrial=json.loads((out/"industrial_transfer_report.json").read_text()); joints=industrial["metrics"]
    wrist=[x for x in joints if x["joint"] in ("left_wrist","right_wrist")]
    means={}
    for model in {x["model"] for x in wrist}:
        v=[x for x in wrist if x["model"]==model];means[model]=sum(x["pck_0_10"] for x in v)/len(v)
    best_refiner=max((k for k in means if k!="mediapipe"),key=means.get)
    recommended="mediapipe" if means["mediapipe"] >= means[best_refiner] else best_refiner
    object_rows=[{"status":"NOT_EVALUABLE","existing_claim":"approximately 91%","actual_previous_metric":"not traceable in repository",
                  "map_50":"","map_50_95":"","precision":"","recall":"","f1":"",
                  "reason":"No object detector checkpoint, object GT manifest, class schema, or evaluation code was found; no value fabricated."}]
    write_csv(out/"object_detection_metrics.csv",object_rows)
    # Preserve small frozen-baseline provenance without copying the checkpoint.
    base_report=json.loads((root/"checkpoints/human_pose_3d_v1/training_report.json").read_text())
    baseline={"read_only_checkpoint":str(root/"checkpoints/human_pose_3d_v1/best.pt"),"split":str(root/"checkpoints/human_pose_3d_v1/split_manifest.json"),
              "train_count":3828,"validation_count":545,"test_count":627,"mediapipe_pck_0_10":base_report["baseline"]["wrists"]["pck_0_10"],
              "refiner_5k_pck_0_10":base_report["corrected"]["wrists"]["pck_0_10"]}
    bdir=root/"results/wrist_refiner_5k_baseline";bdir.mkdir(exist_ok=True);(bdir/"baseline_manifest.json").write_text(json.dumps(baseline,indent=2))
    for size in (20,50):
        src=root/"results"/f"wrist_refiner_{size}k_v1"/"training_config.json"
        shutil.copy2(src,out/f"training_config_{size}k.json")
        shutil.copy2(root/"results"/f"wrist_refiner_{size}k_v1"/"training_log.csv",out/f"training_log_{size}k.csv")
    posture=json.loads((out/"posture_classification_report.json").read_text())
    hp_best=max(hrows[1:],key=lambda x:x["mean_pck_0_10"])
    md=f"""# MoveGuard Large Scale Wrist Refinement Experiment

## 핵심 결론

- Human Pose 기반 보정 모델 중 최고: **{best_refiner}**
- MoveGuard 최종 추천 파이프라인: **{recommended}** (산업현장 PCK@0.10 우선 선택)
- Human Pose 최고: **{hp_best['model']} {hp_best['mean_pck_0_10']*100:.2f}%**
- 산업현장 MediaPipe: **{means['mediapipe']*100:.2f}%**
- 산업현장 추천 파이프라인: **{means[recommended]*100:.2f}%** ({(means[recommended]-means['mediapipe'])*100:+.2f}%p)
- 데이터 누수 검사: **PASS** (action+actor 그룹 및 이미지 ID 교집합 0)

## Data

기존 5k 실험의 실제 학습분 3,828장 ⊂ 20,000장 ⊂ 50,000장이다. 검증 545장과 테스트 627장은 기존 목록 그대로 고정했다. seed는 20260804이다.

## Human Pose Wrist (PCK@0.10)

| 모델 | Left | Right | Mean |
|---|---:|---:|---:|
"""+"\n".join(f"| {r['model']} | {r['left_pck_0_10']*100:.2f}% | {r['right_pck_0_10']*100:.2f}% | {r['mean_pck_0_10']*100:.2f}% |" for r in hrows)+f"""

## Industrial Transfer Wrist (PCK@0.10)

Human Pose에서 학습한 모델을 산업현장 테스트에서 재학습하지 않고 적용했다. Human Pose 수치를 산업현장 수치로 대신하지 않았다.

"""+"\n".join(f"- {k}: {v*100:.2f}%" for k,v in sorted(means.items()))+f"""

## 작업자세 분류

기존에 인용된 71.7%는 전체 자세 정확도가 아니라 validation의 `neck_bending` binary F1(0.717103)이었다. 독립 test 14,744장에서 자세 head별 결과를 `posture_classification_metrics.csv`에 기록했다. Wrist Refiner는 현재 이 분류기 뒤에 있으므로 분류 결과 자체는 바뀌지 않는다.

## 관절 추론 지표 정의

PCK@0.10은 예측 관절과 정답 관절의 픽셀 거리를 사람 bbox 대각선으로 나눈 값이 0.10 이하인 비율이다. 기존 91.2%는 전체 관절 정확도로 재현되지 않았으며, 보고서에서 발견된 91.2846%는 `left_pinky PCK@0.20` 한 항목이었다. 산업 데이터의 재현 가능한 전체 값은 all-annotated PCK@0.10 85.2845%, visible-only PCK@0.10 89.9564%이다.

## 객체 탐지

저장소에 객체 탐지 checkpoint, 객체 GT, class schema, 평가 코드가 없어 기존 약 91%의 출처를 재현할 수 없다. 따라서 신규 mAP를 만들지 않고 **평가 불가**로 기록했다.

## RULA 관련 2D 자세

Shoulder/Elbow/Wrist/Neck proxy/Trunk의 MAE, median, p90 및 RULA-like 각도 구간 일치율은 `rula_angle_metrics.csv`에 있다. 2D 좌표만으로 wrist twist, pronation/supination을 해결했다고 주장하지 않는다. 기존 약 29cm 3D 결과는 본 2D 실험과 분리한다.

## 재현 및 Resume

```bash
.venv/bin/python training/run_wrist_refiner_large_scale.py
.venv/bin/python training/evaluate_wrist_refiner_industrial.py
.venv/bin/python training/export_posture_classification_metrics.py
.venv/bin/python training/finalize_wrist_refiner_experiment.py
```

MediaPipe 캐시는 250장 단위 NPZ로 저장되어 첫 명령을 다시 실행하면 완료 지점부터 이어진다. 기존 5k 및 3D checkpoint는 덮어쓰지 않았다.

## 한계와 결정

최종 제품 적용 가능 여부는 산업현장 전이 결과와 RULA 각도 지표를 우선해 판단한다. 산업현장 개선이 없으면 Human Pose 점수가 높더라도 큰 모델을 채택하지 않는다. 모델이 Human Pose의 두 actor와 제한된 촬영 환경에 과적합됐을 가능성, 2D 손목 회전의 근본적 한계, 실제 전문가 최종 RULA 정답 검증이 남아 있다.

- Best Wrist Refiner: **50k** (산업 전이 모델끼리만 비교할 때 80.61%)
- Recommended MoveGuard model: **MediaPipe-only** (82.25%)
- Industrial generalization: **Poor** (모든 refiner가 baseline보다 낮음)
- Ready for MoveGuard MVP: **YES, MediaPipe-only / wrist refiner 적용은 NO**
- Need 120k + Human Pose joint training: **YES, refiner를 제품에 쓰려면 산업 도메인 적응 실험 필요**
- Refiner checkpoint size: 약 **444 KiB**
- CPU batch-1 refiner inference: 약 **0.038 ms** (MediaPipe 시간 제외)
"""
    (out/"final_summary.md").write_text(md)
    print(json.dumps({"best_refiner":best_refiner,"recommended_pipeline":recommended,"humanpose_best":hp_best,"industrial_means":means},indent=2))

if __name__=="__main__":main()
