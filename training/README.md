# AI-Hub 자세 분류 파이프라인

원본 ZIP은 읽기 전용으로 스트리밍하며 수정·이동·삭제하거나 전체 압축 해제하지 않는다.

```bash
python training/aihub_audit.py \
  --data-root "/absolute/path/01-1.정식개방데이터" \
  --output artifacts/aihub/audit \
  --seed 20260803
```

`split_manifest.jsonl`은 파일명 끝의 프레임 번호를 제거한 촬영 시퀀스를 그룹으로
묶어 70/15/15로 분리한다. 동일 작업자·영상·연속 프레임이 서로 다른 split에
섞이지 않도록 이후 전처리와 학습은 이 manifest만 사용한다.

1,000장 사전 점검(같은 명령을 재실행하면 checkpoint에서 이어진다):

```bash
.venv/bin/python training/preprocess_mediapipe.py \
  --manifest artifacts/aihub/audit/split_manifest.jsonl \
  --output artifacts/aihub/pilot --limit 1000
```
