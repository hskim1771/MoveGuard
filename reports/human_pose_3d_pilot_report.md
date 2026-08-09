# Human Pose 3D pilot

- Branch: `experiment/human-pose-3d`
- Seed: `20260804`
- Source: AI-Hub datasetkey 209 validation, 5,000 selected images
- Split: actor + action grouped 70/15/15 (3,828 / 545 / 627 images)
- Group overlap: 0
- Model: MediaPipe Pose 33 landmarks followed by a 2-layer residual MLP corrector
- Test detections: 615 / 627 (98.09%)

## Independent test results

| Scope | Metric | MediaPipe mapping | Corrected | Change |
|---|---:|---:|---:|---:|
| All 24 joints | Mean pixel error | 25.12 px | 15.53 px | -38.18% |
| All 24 joints | PCK@0.05 | 66.65% | 83.22% | +16.57%p |
| All 24 joints | PCK@0.10 | 90.54% | 94.66% | +4.13%p |
| Wrists | Mean pixel error | 39.06 px | 17.33 px | -55.64% |
| Wrists | PCK@0.05 | 28.62% | 81.87% | +53.25%p |
| Wrists | PCK@0.10 | 78.78% | 93.25% | +14.47%p |

PCK is normalized by the full annotated pose bounding-box diagonal. The test split
was not used for model or epoch selection.

## Runtime

- MediaPipe preprocessing: 5,000 images in 155.5 seconds
- Corrector checkpoint: 444 KiB
- Corrector CPU latency: approximately 0.038 ms/person on this Mac
- Accelerator used for training: Apple MPS

## Limits

- This pilot evaluates 2-D keypoint correction, not a retrained MediaPipe model.
- It contains two validation actors and therefore does not establish real-world generalization.
- The 3-D `3d_rot` vector exists, but its components are not explicitly named roll/pitch/yaw in the JSON.
- A separate 3-D coordinate experiment is required before claiming 3-D wrist accuracy.
- Camera footage and additional workers must be evaluated before production deployment.

## Wrist rotation pilot

The 3,819 image-to-3D-label links were evaluated separately. Actor + action groups
were kept wholly inside train, validation, or test (2,726 / 519 / 574 images).
MediaPipe normalized and world landmarks were used to predict the left and right
`3d_rot` three-component vectors.

| Model | Mean circular error | Within 5 | Within 10 | Within 15 |
|---|---:|---:|---:|---:|
| Training-set mean | 4.18 | 72.97% | 87.77% | 95.39% |
| Ridge | 4.86 | 66.23% | 86.02% | 94.75% |
| MLP | 4.44 | 68.56% | 88.50% | 95.77% |

The MLP did not beat the constant baseline on mean error and improved the within-10
rate by only 0.73 percentage points. It is therefore rejected for production RULA
wrist scoring. AI-Hub describes these values as roll, pitch, and yaw, but the JSON
does not explicitly name the order of the three array elements. Axis semantics and
neutral-angle calibration must be confirmed before flexion/extension or twist scores
can be derived.
