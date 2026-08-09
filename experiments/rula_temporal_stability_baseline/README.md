# RULA temporal stability baseline

- 기준 브랜치: `feature/figma-mediapipe-joint-domain-integration`
- 기준 커밋: `605a5fa468d7c575ff82a780a3e2106ab28a8088`
- 기준 방법: `mediapipe_raw`, `joint_raw`
- 비교 방법: 각 raw 방법에 동일한 `TemporalPoseStabilizer`를 적용한 stabilized 방법
- 독립 테스트 원본 결과: `../wrist_rula_pipeline_validation/validation_report.json`
- 프레임별 결과: `../wrist_rula_pipeline_validation/frame_metrics.jsonl` (로컬 전용, Git 제외)
- 그룹별 최종 RULA 비교: `../wrist_rula_pipeline_validation/group_rula_comparison.csv`

원시 결과를 덮어쓰지 않았다. 검증셋과 독립 테스트셋 결과는 서로 다른 디렉터리에 저장했다.
