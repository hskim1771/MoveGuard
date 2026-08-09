# Multimodal RULA posture model

`multimodal_rula_v1.pt` is MoveGuard's lightweight image + MediaPipe feature
multi-head posture model. It predicts AI-Hub posture labels used as inputs to
the deterministic RULA Table A/B/C implementation; it does not replace the
tables or claim expert final-RULA accuracy.

- Training frames: 66,250
- Validation frames: 13,782
- Independent test frames: 14,744
- Seed: 20260803
- SHA-256: `16af4dc91e3afd01ab4ceacdf561c47e3abbff3a75d7359bec65b768a9d3fc74`
- Full metrics and limitations: `reports/aihub_full_120k_training_report.md`
