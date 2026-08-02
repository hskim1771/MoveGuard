# Posture analysis v2 architecture

```text
image frame
  -> PoseEstimator
  -> wrist ROI -> HandEstimator
  -> ObjectDetector (optional)
  -> angle_calculator
  -> rula_engine
  -> API response / later persistence
```

## Module boundaries

- `estimators/base.py`: 모델 공통 인터페이스
- `estimators/mediapipe_pose.py`: 현재 전신 모델 어댑터
- `estimators/mediapipe_hand.py`: 손목 ROI별 21개 손 관절 어댑터
- `estimators/rtmw_pose.py`: 공식 RTMW-l 384×288 whole-body ONNX 어댑터
- `joint_mapper.py`: 모델별 출력을 공통 관절명과 source로 정규화
- `estimators/object_detector.py`: 명시적 미설정 어댑터와 선택형 YOLO 어댑터
- `angle_calculator.py`: 모델과 분리된 관절 각도 계산
- `rula_engine.py`: 각도와 RULA 조견표 계산
- `pipeline.py`: 사람·손·객체 연결 및 감사 가능한 JSON 결과 조립
- `api.py`: 이미지, multipart, landmark JSON 입력 처리

## Model replacement

현재 `baseline`과 `rtmw`를 `estimators/factory.py`에서 선택할 수 있습니다.
두 모델은 같은 `angle_calculator`와 `rula_engine`을 사용합니다. RTMW-l의
133개 keypoint는 MediaPipe 호환 33-slot과 공통 관절명으로 변환되고, 손목
각도에는 같은 whole-body 출력의 손 keypoint를 사용합니다.

향후 모델도 `estimators/base.py`의 인터페이스를 구현하고 factory 분기만
추가합니다.

- 다른 RTMW/RTMPose: `PoseEstimator` 구현으로 추가
- RTMPose Hand5 또는 HaMeR: `HandEstimator` 구현으로 추가
- 다른 YOLO/RTMDet: `ObjectDetector` 구현으로 추가
- MotionBERT: Pose 결과 뒤의 temporal/3D 보정 단계로 추가

이 경우 API 응답과 `angle_calculator`, `rula_engine`은 변경할 필요가 없습니다.

## Known limitations

- 단일 RGB 카메라는 손목 회내·회외와 물체 하중을 신뢰성 있게 직접 측정하지
  못합니다. 손목 twist 수동값이 없으면 중간 범위를 가정하고 응답에 가정을
  기록합니다.
- 현재 RULA lookup은 기존 MoveGuard MVP 조견표와 비교 가능하도록 동일한
  단순화 표를 사용합니다. 의료 진단이나 법적 판정용이 아닙니다.
- 다리 지지는 무릎 굽힘과 좌우 대칭으로 추론합니다. 실제 발 지지면과 하중은
  현장 입력이나 추가 센서로 확인해야 합니다.
- 공개 모델의 AP는 MoveGuard RULA 정확도가 아닙니다. 전문가 정답 데이터로
  최종 점수 일치율과 ±1 일치율을 별도로 평가해야 합니다.
