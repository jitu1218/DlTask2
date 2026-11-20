# Custom CNN for Road Sign Classification and Occlusion Detection

A PyTorch assignment with a custom shared CNN and two output heads:
43-class traffic sign classification and binary synthetic occlusion detection.
Includes real GTSRB subset training results, a checkpoint, plots, metrics,
unit tests, and commands for a full-data experiment.

## Run on Windows

Use Python 3.11 or 3.12. Extract the project and open PowerShell inside
`road_sign_occlusion`, where `train.py` is located.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m unittest discover -s tests -v
```

For macOS/Linux replace `.venv\Scripts\python` with `.venv/bin/python`.
The scripts use CPU intentionally for portability. No API key is required.
Dependency installation and the first dataset download need internet. If pip
reports a DNS error resolving pypi.org, fix the connection and retry.

### See the completed run immediately

Open `demo_results/REPORT.html` in a browser. This is an actual, limited-data
experiment, not a claim of full-benchmark or road-ready performance.
Its configuration and exact results are recorded beside the report.

### Reproduce the included subset run

```powershell
.venv\Scripts\python train.py --download --epochs 12 --train-per-class 100 --val-per-class 30 --test-per-class 40 --output my_demo_results
```

### Train using all available images

```powershell
.venv\Scripts\python train.py --download --epochs 20 --output results
```

The three official archives total roughly 365 MB compressed; allow additional
space for extracted images. Downloads are stored under `data/` and reused.
Training time depends on the CPU; the full run is substantially slower than the
subset demonstration. `--threads 2` is the default; adjust to suit your machine.

If automated downloading fails, download these archives from the official archive
and place them under `data/`, then rerun with `--download` to extract:

- `GTSRB_Final_Training_Images.zip`
- `GTSRB_Final_Test_Images.zip`
- `GTSRB_Final_Test_GT.zip`

Official archive:
https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html

Archives, raw dataset images and dependencies are not bundled in this ZIP.
If an existing archive is corrupted, delete that archive and rerun the download.

### Predict on a sign crop

```powershell
.venv\Scripts\python predict.py --checkpoint demo_results/model.pt --image examples\occluded_sign.png
```

The image should contain one tightly cropped sign. Output includes the GTSRB
class ID (0–42), an uncalibrated sign softmax score, occlusion sigmoid score,
validation-selected threshold, and the binary prediction. The application does
not find signs in a full road image; that would require a separate object detector.

## Dataset and leakage prevention

GTSRB is a public benchmark of real traffic sign photographs with 43 categories.
There are 39,209 official final training images and 12,630 official final test
images. Training images are grouped into tracks of the same physical sign.

1. Load official semicolon-delimited annotations and sign bounding boxes.
2. Split the training tracks, separately within each class, into roughly 80%
   training and 20% validation. A class + filename track prefix identifies each
   group. Consecutive frames cannot cross the split boundary.
3. Apply any requested per-class subset limits only after the group split.
4. Crop to the annotated region, resize to 64×64, normalize pixels to [-1, 1].
5. Add random occluders only after assigning data to subsets.
6. Keep official test images separate from training and validation.

Full training/validation counts for seed 42: 31,380 / 7,829 original images.
Evaluation creates two views of each original: unchanged and synthetically
occluded. The resulting examples are balanced 50/50 but paired and correlated.

## Occlusion generation

`occlusion.py` generates three types of procedural overlays:

- Leaves: overlapping green ellipses with textured color variation.
- Mud: irregular brown blobs with texture.
- Graffiti: thick randomly colored strokes.

Placement, scale and appearance vary. The measured binary mask coverage is
recorded with each evaluated positive. It is crop coverage, not a true sign
segmentation measurement. Training has a 50% chance of adding an overlay;
validation/test overlays are fixed by their seeds. No horizontal flips are used.

These overlays approximate appearances; they are not photorealistic. The label
means **added synthetic occlusion**. An unchanged GTSRB image is not guaranteed
to be naturally unobstructed, since this project does not have natural-occlusion
annotations. Natural occlusion recognition remains an external validation task.

## Network and training

- Three Conv2d → ReLU → MaxPool blocks with 16, 32, 64 channels.
- Adaptive pooling to 4×4; shared dense layer with 128 units and dropout 0.25.
- Sign head: 43 logits, trained using cross-entropy.
- Occlusion head: one logit, trained using binary cross-entropy with logits.
- Joint objective: sign CE + occlusion BCE (equal task weights).
- AdamW: learning rate 0.001, weight decay 0.0001, batch size 64.
- Early stopping: patience 5, minimum improvement 0.0001 in validation joint loss.
- Restore the best model; choose occlusion threshold by maximum validation F1
  over 0.05–0.95 in steps of 0.01; lock both before evaluating the test set.

Training loss includes active dropout and changing augmentations; validation loss
uses evaluation mode and fixed views. They are useful learning indicators, but
are not exactly the same data distribution.

## Evaluation

Outputs include occlusion precision, recall, F1, average precision, confusion
matrix, precision-recall curve, recall by occluder and coverage band, and sign
accuracy separately on clean and occluded images.

Precision = TP / (TP + FP). Recall = TP / (TP + FN).
A false negative is a missed synthetic occlusion. Higher recall can be obtained
by lowering the threshold, potentially increasing false alarms. This project
chooses F1 on validation; it does not invent a safety-critical operating target.

Balanced synthetic evaluation does not represent real-world occlusion prevalence.
Do not use test metrics for repeated tuning and then claim an untouched test
estimate. Further development requires fresh real-world test data, natural
occlusion labels, and testing on occluder types absent from training.

## Project files

| File | Role |
|---|---|
| `model.py` | Custom dual-head CNN |
| `data.py` | Download, annotations, ROI crops, group splits and datasets |
| `occlusion.py` | Synthetic occluder generation |
| `train.py` | Training, checkpoint selection, thresholding, metrics |
| `predict.py` | Inference on one sign crop |
| `report.py` | Figures and Markdown/HTML report generation |
| `tests/test_suite.py` | Five tests covering core correctness |
| `demo_results/` | Completed subset run and saved checkpoint |

Each result folder contains `config.json`, `split_manifest.json`,
`selection.json`, `model.pt`, `metrics.json`, `history.csv`,
`test_predictions.csv`, figures and reports. Manifest paths describe the machine
that generated a run; rerunning generates paths appropriate for your machine.

## Submission and explanation

Viva summary: “My CNN shares visual features between a sign-classification head
and an occlusion head. I synthesize three occluder types, separate physical-sign
tracks before augmentation, and evaluate occlusion precision and recall on held-out
images. My demonstration measures synthetic occlusion on sign crops, not autonomous
driving readiness.”

References:
- https://benchmark.ini.rub.de/
- https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html
- https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
