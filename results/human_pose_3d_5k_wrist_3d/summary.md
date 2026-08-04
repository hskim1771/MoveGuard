# MoveGuard Human Pose 3D 5k Wrist Experiment

## Scope and preserved baseline

- Selected RGB images: 5,000
- Image-to-3D-label links used: 3,819
- Train / validation / test: 2,813 / 379 / 627
- MediaPipe detections: 2,757 / 378 / 615
- Seed: 20260804
- Split: the original 2-D actor + action group manifest was reused; overlap is zero.
- Existing 2-D model/checkpoint and its PCK@0.10 result were not overwritten.

## Meaning of the existing 93.25%

It is wrist PCK@0.10, not detection rate. For each detected test image, the
Euclidean 2-D error of left and right wrist is divided by the diagonal of the
full 24-joint ground-truth pose bounding box. A wrist is correct when the ratio
is at most 0.10. The reported value combines left and right wrists:

`correct left/right wrists / (detected test images × 2) = 93.252%`.

PCK@0.05 is 81.870%. MediaPipe detection rate is reported separately as
615/627 = 98.086%. Counting detection failures as incorrect gives an end-to-end
PCK@0.10 of 91.467%.

## JSON and coordinate interpretation

- `info.3d_pos`: ordered names for 24 joints.
- `annotations.3d_pos`: 24 × 4 homogeneous positions; the first three values
  are x/y/z and the fourth is 1.0.
- `info.3d_rot`: the same 24 joint names.
- `annotations.3d_rot`: 24 × 3 per-joint rotation vectors.
- `annotations.trans_params`: separate body/global translation.
- `annotations.obj_path`: corresponding 3-D body mesh path.

The 3-D label is shared by multiple camera images of the same actor/action/frame,
so it is not a per-camera pixel+depth coordinate. Pelvis is non-zero, so it is
not root-relative. Across 1,000 labels, skeleton median span is approximately
710 × 1,349 × 283 coordinate units. Pelvis x/z also closely follows
`trans_params × 1000`. This strongly indicates a global capture/body coordinate
in millimetres, although the downloaded JSON does not explicitly state the unit.
Z is the global/capture depth axis, not MediaPipe image-relative z.

MediaPipe world landmarks are pelvis-centred, meter-scale and camera/model-axis
coordinates. Their raw values therefore cannot be directly subtracted from the
AI-Hub global coordinates. Raw, root-aligned, root+scale-aligned, Procrustes, and
learned-model metrics are kept separate.

## Independent test metrics

The new MLP uses all 33 MediaPipe normalized landmarks and all 33 world landmarks
as full-body context, and predicts all 24 AI-Hub joints in 3-D. Test data was not
used for model/epoch selection.

| Metric | Left wrist | Right wrist | Mean wrist |
|---|---:|---:|---:|
| MLP MPJPE | 277.38 | 304.62 | 291.00 |
| Ridge MPJPE | 307.45 | 329.87 | 318.66 |
| Train-mean MPJPE | 313.02 | 343.19 | 328.11 |
| MediaPipe raw (incompatible coordinates) | 1123.99 | 1210.63 | 1167.31 |
| MediaPipe root-aligned | 686.35 | 568.07 | 627.21 |
| MediaPipe Procrustes-aligned (diagnostic/oracle) | 321.56 | 331.93 | 326.74 |

MLP axis MAE:

- Left wrist: X 134.29, Y 186.18, Z 99.90
- Right wrist: X 140.60, Y 206.45, Z 105.80

Values are in the AI-Hub JSON coordinate unit; the evidence above strongly
supports interpreting them as millimetres. The approximately 291 mm wrist error
is not suitable for production 3-D wrist scoring.

## View and posture findings

Camera calibration metadata was not downloaded, so camera numbers cannot be
honestly renamed front/side/diagonal. Per-camera results are in
`per_view_metrics.csv`. Best mean wrist MPJPE was camera 8 (243.05), followed by
camera 5 (243.56); worst was camera 6 (378.52), followed by camera 7 (329.70).

- Arms up: 405.48 mean wrist MPJPE
- Arms down: 282.82
- Elbow flexed: 296.96
- Elbow extended: 290.79
- Wrist near body: 293.13
- Wrist far from body: 286.95

## Worst 20

The worst 20 are listed in `worst_cases.csv`. These are conservative automatic
diagnostic signals, not expert visual ground truth:

- Camera distribution: camera 6 = 8, camera 7 = 7, camera 1 = 5
- Action distribution: action 42 = 19, action 35 = 1
- Low MediaPipe wrist visibility: 10/20
- Possible blur: 6/20
- Z/depth-dominant error: 13/20
- Arm/body overlap signal: 20/20

## Rotation and RULA feasibility

- Wrist position: possible to train, but current 3-D error is too high for deployment.
- Flexion/extension: conditionally possible from elbow-wrist-hand 3-D geometry;
  current position error must improve first.
- Radial/ulnar deviation: conditionally possible with a validated hand axis;
  `L_Hand/R_Hand` are single points and no detailed fingers are provided in 3-D.
- Pronation/supination: not reliably obtainable from wrist position alone. It
  requires a hand/forearm orientation or validated per-joint rotation convention.

`3d_rot` is structurally per-joint because its 24 entries align with the 24 names
in `info.3d_rot`; it is not camera rotation or the separate whole-body translation.
However, the JSON does not explicitly document Euler order, local/global basis,
rotation convention, or units. It must not yet be converted into RULA wrist twist.

## Recommendation

- Human Pose 3D 5k utility for MoveGuard: **medium**. It clearly improves 2-D
  wrist localization but does not yet provide reliable global 3-D wrists.
- Combining with the industrial 120k model: **conditionally recommended**, only
  after camera calibration/coordinate mapping and a multi-actor 3-D validation.
- Do not mix-train yet. Main risks are domain gap, coordinate/keypoint mismatch,
  industrial-pose regression, and catastrophic forgetting.
