#!/usr/bin/env python3
"""Small terminal dashboard for the resumable large-scale wrist experiment."""
import json
import os
import time
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "results/wrist_refiner_large_scale_comparison/progress.json"
while True:
    os.system("clear")
    print("MoveGuard 손목 보정 대규모 실험\n")
    if path.exists():
        try:
            p = json.loads(path.read_text())
            done, total = p.get("processed", 0), p.get("total", 0)
            percent = p.get("percent", 0)
            width = 40
            filled = int(width * percent / 100)
            print(f"단계: {p.get('phase', '-')}")
            print("[" + "#" * filled + "-" * (width - filled) + f"] {percent:.2f}%")
            print(f"진행: {done:,} / {total:,}")
            print(f"속도: {p.get('items_per_second', 0):,}/초")
            eta = p.get("eta_seconds")
            print(f"예상 남은 시간: {eta / 60:.1f}분" if eta is not None else "예상 남은 시간: 계산 중")
            print(f"마지막 갱신: {p.get('updated_at', '-')}")
            for key in ("matched", "train_loss", "val_loss", "best_val_loss", "patience"):
                if key in p:
                    print(f"{key}: {p[key]}")
        except Exception as exc:
            print(f"진행 파일 갱신 중: {exc}")
    else:
        print("실험 시작을 기다리는 중...")
    time.sleep(2)
