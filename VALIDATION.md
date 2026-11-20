# Validation record

- Downloaded and parsed official GTSRB final training images, test images and test labels.
- Completed 12 epochs on a capped subset (up to 100 training, 30 validation and 40 test original images per class). See demo_results/config.json for exact counts.
- Evaluated 3,440 paired test views from 1,720 original test images after locking the validation-selected checkpoint and threshold.
- All five unit tests passed: dual-head gradient flow, deterministic/non-mutating overlays, track-disjoint splits, fixed evaluation pairs, and threshold/metric correctness.
- Successfully loaded the checkpoint and ran predict.py on examples/occluded_sign.png.
- Inspected synthetic-occlusion examples; generated learning and precision-recall plots, confusion matrix, and HTML/Markdown reports.
- Python source formatted with Black. Executed on Linux CPU; Windows commands documented but not run on Windows.
- Full-data training was not executed. This is a synthetic-occlusion subset demonstration, not a real-world safety validation.
