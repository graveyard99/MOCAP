# Official PPHumanSeg integration and acquisition

Status: the deployment adapter and generated-score runtime tests are WORKING. Actual checkpoint acquisition and real-image inference remain **BLOCKED_EXTERNAL** in this execution environment. Network execution requests stalled before execution and were aborted; default sandbox network cannot reach the proxy. No model binary was transferred, no real-person mask was generated and no local checkpoint inference is claimed.

## Publisher, terms and immutable artifact identity

The official [OpenCV Zoo model README](https://github.com/opencv/opencv_zoo/tree/main/models/human_segmentation_pphumanseg) states that the files in this model directory use Apache-2.0. The directory [license](https://github.com/opencv/opencv_zoo/blob/main/models/human_segmentation_pphumanseg/LICENSE) attributes PaddlePaddle Authors (2021). These official pages and the implementation were inspected through the web reader on 2026-10-06. The model originates from PaddleHub. No gated account, body asset or credential is needed for this public file.

The float32 artifact is `human_segmentation_pphumanseg_2023mar.onnx`. Its [publisher Git LFS pointer](https://github.com/opencv/opencv_zoo/blob/main/models/human_segmentation_pphumanseg/human_segmentation_pphumanseg_2023mar.onnx) identifies exactly 6,163,938 bytes and SHA-256:

```text
552d8a984054e59b5d773d24b9b12022b22046ceb2bbc4c9aaeaceb36a9ddf24
```

These are publisher values, not an observed local checksum. Downloading the raw GitHub source URL returns the tiny LFS pointer, not the ONNX binary. Use the official media URL below. No checkpoint is committed or redistributed by this repository.

## Download on a desktop with working network access

The shared acquisition helper now includes detector, pose and this segmentation asset. `python scripts/fetch_models.py --asset segmentation --timeout 40` downloads only PPHumanSeg, while `--asset all` selects all three. It verifies the published PPHumanSeg size/SHA-256 before installation and leaves inference verification false until actual execution. The command was not executed successfully in this environment. Manual acquisition remains possible from the official media URL or using the bounded commands below. Set the actual installation path first; this example never changes the user's HOME.

```bash
INSTALL_ROOT=/absolute/path/to/openmocap-install
mkdir -p "$INSTALL_ROOT/downloads" "$INSTALL_ROOT/models/segmentation"
curl --fail --location --connect-timeout 10 --max-time 40 \
  --output "$INSTALL_ROOT/downloads/human_segmentation_pphumanseg_2023mar.onnx.partial" \
  https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/human_segmentation_pphumanseg/human_segmentation_pphumanseg_2023mar.onnx
export INSTALL_ROOT
python - <<'PY_CHECK'
import hashlib, os
from pathlib import Path
root = Path(os.environ['INSTALL_ROOT'])
source = root / 'downloads/human_segmentation_pphumanseg_2023mar.onnx.partial'
expected = '552d8a984054e59b5d773d24b9b12022b22046ceb2bbc4c9aaeaceb36a9ddf24'
assert source.stat().st_size == 6163938, 'Publisher size mismatch; do not install'
assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, 'Publisher checksum mismatch'
source.replace(root / 'models/segmentation/human_segmentation_pphumanseg_2023mar.onnx')
print('Publisher size/SHA256 verified; model installed locally')
PY_CHECK
```

Retain the publisher license/attribution when redistributing the asset. The supplied directory terms do not establish accuracy on a particular capture. This system still prioritizes reliable measured joints over body-envelope inference.

## Exact inspected deployment contract

The [publisher implementation](https://github.com/opencv/opencv_zoo/blob/main/models/human_segmentation_pphumanseg/pphumanseg.py) converts decoded BGR plates to RGB, resizes directly to 192 × 192, scales bytes to [0,1], subtracts mean [0.5,0.5,0.5], divides by standard deviation [0.5,0.5,0.5] and supplies float32 NCHW. ImageNet means/standard deviations are incorrect for this checkpoint.

The publisher resizes the two output class-score planes to the original plate and then selects argmax, with foreground class 1. `output_format: pphumanseg` matches that ordering and keeps ties in background. `logits: null` detects whether scores are already normalized probabilities; otherwise softmax is used only for a heuristic confidence. Argmax mask selection is preserved. Confidence is not a calibrated segmentation-accuracy probability. Instance assignment clips the semantic result to a person's box; this is not a learned instance-segmentation system.

[configs/segmentation/pphumanseg.yaml](../../configs/segmentation/pphumanseg.yaml)
is a portable inference preset. After installing the checkpoint, open Actor →
Inference checkpoints → Load installed whole-body preset… and choose this file.
The GUI and shared perception service resolve `${INSTALL_ROOT}`; no manual path
edit is required. The importer preserves existing detector/pose settings. An
explicit absolute local path also works. The normal image/box input and mask
output APIs match the other segmentation backends.

## Verification completed and remaining

Six offline tests execute generated Constant-score ONNX models in actual CPU ONNX Runtime and verify RGB/float32/normalization, resize-before-argmax, ties, probability-versus-score confidence, ROI bounds, invalid contracts and template options. Together with existing perception tests, the focused run passed 15 tests in 0.37 seconds. These original numerical fixtures are not the PPHumanSeg checkpoint.

After acquiring the file, run the segment stage on authorized real media and inspect masks/confidence in saved raw outputs; record the observed model hash, runtime/provider, output shape, selected image and mask coverage. Actual learned-model execution, real-image quality and GPU performance remain unverified here. Silhouette-based actor-shape fitting remains **NOT_STARTED**; adding this segmentation checkpoint does not make masks affect the fitted body.
