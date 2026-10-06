# Public whole-body inference assets

The installed inference preset is `configs/pose/rtmpose-wholebody-yolox.yaml`.
The desktop application's **Load inference preset** action expands `${INSTALL_ROOT}`
to the installation root and validates model files before applying settings.
The same settings are consumed by the shared perception pipeline API.

## Acquisition and isolation

Use the isolated environment:

```bash
source /workspace/openmocap-install/openmocap-vfx/scripts/activate.sh
python /workspace/openmocap-install/openmocap-vfx/scripts/fetch_models.py --asset all
```

The command downloads from the official OpenMMLab host and OpenCV Zoo media URL. Archives stay
in `INSTALL_ROOT/downloads`; weights, deployment metadata and manifests stay in
`INSTALL_ROOT/models`. Requests have connect and total timeouts. Only one
identified ONNX model and small deployment JSON files are extracted; scripts in
archives are never executed. Weights and archives are excluded from Git.
A repeat command verifies the recorded model hash before reusing an installation.
An interrupted download cannot become a completed model file.

| Component | Public asset | Installed model |
|---|---|---|
| Person detection | YOLOX-m, HumanArt, `c2c7a14a` | `models/detection/yolox_m_humanart.onnx` |
| Whole-body pose | RTMW-l distilled, Cocktail14, 2023-11-22, 256×192 | `models/pose/rtmw_wholebody133.onnx` |
| Human segmentation | OpenCV Zoo PPHumanSeg, 2023-03, 192×192 | `models/segmentation/human_segmentation_pphumanseg_2023mar.onnx` |

`--asset all` selects these three public assets, and excludes licensed body
models. For PPHumanSeg, the installer verifies the publisher's Git LFS SHA-256
and file size. Load its separate `configs/segmentation/pphumanseg.yaml` preset
after the whole-body preset to add the matching mask deployment settings.
See [PPHumanSeg contracts and terms](../segmentation/public-model.md).

RTMW-l is selected because an author-published whole-body ONNX export is available,
it retains all 133 COCO-WholeBody landmarks, and it runs with the existing CPU
ONNX Runtime environment without PyTorch, MMCV or a proprietary runtime.
This choice does not establish that it is the most accurate model for your plates.
The 384×288 model is an accuracy-oriented upstream alternative; its different
asset, input size and runtime costs need a separate verified preset.

## Exact input/output semantics

The preset never guesses a model's output format. Detection uses top-left BGR
letterboxing at the graph's configured input resolution, with fill value 114,
float32 NCHW, and no /255 normalization. The explicit `mmdeploy` codec consumes
XYXY+score detection tensors and a separate class-label tensor. Exports with raw
YOLOX heads must select `yolox_raw`; its configured 8/16/32 stride grids decode
centres and log widths/heights before person-score filtering and NMS.

Pose inference uses a reversible top-down affine crop with padding 1.25,
fixed input aspect, BGR input, and the official example's channel-order
normalization: mean `[123.675,116.28,103.53]`, standard deviation
`[58.395,57.12,57.375]` on the 0–255 range. These deployment assets' reference
example does **not** insert an additional BGR→RGB conversion.
SimCC maxima are decoded with split ratio 2. The confidence score uses the
minimum of the two axis maxima, conservatively following the official MMPose
ONNX example. A score is a detector/model quality signal, not calibrated
probability of millimetre accuracy.

The shared `openmocap.pose2d.schemas.COCO_WHOLEBODY_133` schema preserves:
17 body landmarks,6 feet landmarks,68 face landmarks and21 landmarks per hand.
These are image landmarks; facial and finger surface landmarks are not
interchangeable with every SMPL-family skeletal joint. The current body solver
fits supported named body joints and does not claim a validated 133-joint
hand/face body-model mapping. Derived hip/shoulder midpoint pelvis/neck values
remain separate, explicitly marked lower-confidence geometric derivations.
Original detector/pose results remain in `raw/pose2d.json`; association writes
derived effective observations without rewriting those raw landmarks.

## Verification

After downloading, run the actual model graph on a legally usable real-person
image, not a generated network-output fixture:

```bash
python /workspace/openmocap-install/openmocap-vfx/scripts/verify_public_models.py /path/to/person.png
```

This validates graph loading, output contracts, person detection, full 133
keypoint decoding and finite plate-space coordinates. It writes an overlay,
raw observations and timings to `INSTALL_ROOT/temp/model-verification`, and
updates each installed manifest with the executed result. A missing detection
fails clearly. This is an inference smoke test, not an accuracy benchmark or a
metric multi-camera reconstruction validation. Check `inference_verification`
in the actual installed manifests for the result; file presence alone is not
proof of successful inference.

For a desktop with a properly installed GPU runtime, the verifier accepts
`--provider CUDAExecutionProvider`. It warms the detector/pose graphs before
timing and records both requested and actual session providers. A requested
GPU provider that falls back to CPU during session creation fails explicitly.
The GPU runtime installer and actual GPU inference remain unvalidated here;
this flag does not install CUDA or prove an accuracy/throughput benchmark.

No public checkpoint was acquired during this build. Direct sandbox requests
failed to reach the proxy, and network-enabled commands stalled before execution
awaiting permission and were aborted. The official graph outputs and the selected
detector deployment codec must still be confirmed against the actual downloaded
models before calling the preset validated.

## Licensing and provenance

Upstream documentation and terms reviewed on 2026-10-06:

- [Official MMPose model zoo](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)
  links the selected RTMW and YOLOX archives on `download.openmmlab.com`.
- [MMPose code licence](https://github.com/open-mmlab/mmpose/blob/main/LICENSE): Apache-2.0.
- [YOLOX code licence](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE): Apache-2.0.
- [DWPose licence](https://github.com/IDEA-Research/DWPose/blob/main/LICENSE): Apache-2.0.
- [HumanArt dataset terms](https://github.com/IDEA-Research/HumanArt): the dataset
  requires form-based authorization for noncommercial use and must not be
  privately transferred. The dataset is not downloaded by these commands.
- Cocktail14 training includes 14 datasets with distinct image/data terms.
  Public author-hosted trained weights and an Apache-licensed inference/code
  repository do not establish an unrestricted right to redistribute every
  training image or provide blanket clearance for commercial model use.

Assets are acquired for the user's explicitly authorized personal noncommercial
experimentation. They are kept outside the repository and are not bundled in
its open-source distribution. No dataset access form, gated body-model account,
SMPL licence or commercial agreement is accepted on the user's behalf.

Manifest SHA256 values are computed from downloaded bytes. They are **observed
local hashes**, not claimed publisher-signed checksums when none is published.
Every manifest records the source URL, size, archive hash, model hash, review
date, exact inference metadata and verification status.
