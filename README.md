# OpenMocap VFX

A geometry-first, metric, asynchronous multi-camera motion-capture engineering preview. Calibrated cameras and timestamped 2D observations produce robust N-view joints, continuous motion, one persistent actor shape, animated local rotations and root motion, floor-aware contacts, QC reports and a genuine rigged/skinned FBX. A local PySide6 desktop application shares the CLI's pipeline services.

**This is not a production-ready release.** The synthetic pipeline and desktop workflow are implemented and tested. Real capture accuracy, licensed SMPL assets, whole-body inference checkpoints and several advanced VFX tools need further validation. See [build status](docs/BUILD_STATUS.md). The explicit synthetic fixture is a technical humanoid, not SMPL and not an inferred human body.

For desktop migration and further development, read [project state and known issues](PROJECT_STATE.md), [agent instructions](AGENTS.md) and [session handoff](SESSION_HANDOFF.md). The handoff source audit records defects beyond the current passing tests; a passing synthetic suite does not establish production readiness.

## Requirements and isolated installation

Linux x86-64 was tested with Python 3.12; Python 3.11–3.13 is supported by package metadata. Geometry runs on CPU. A desktop/display server is needed for interactive Qt. Blender is needed for FBX and FFmpeg/ffprobe for native media timestamps. Existing OS Python, Blender and FFmpeg are detected, not installed or modified. GPU drivers and OS display libraries remain system prerequisites.

The repository is [graveyard99/MOCAP](https://github.com/graveyard99/MOCAP). The original cloud installation is `/workspace/openmocap-install/openmocap-vfx`, with environment `/workspace/openmocap-install/env`. On a local machine choose a writable absolute installation root and clone into its **`openmocap-vfx`** child explicitly; the scripts resolve that directory's parent automatically. The default Git clone directory name `MOCAP` does not match the generated launcher's assumptions. Recreate the environment rather than copying the cloud environment.

```bash
git clone https://github.com/graveyard99/MOCAP.git /absolute/local/root/openmocap-vfx
cd /absolute/local/root/openmocap-vfx
./scripts/bootstrap.sh
source ./scripts/activate.sh
openmocap doctor
"$OPENMOCAP_INSTALL_ROOT/OpenMocap.sh"
```

Bootstrap is idempotent and uses the committed `uv.lock` with `uv sync --frozen`. Python packages, uv, models, downloads, temporary files and library caches remain beneath the installation root. It never installs Python packages globally. It may access PyPI to acquire the pinned bootstrap uv and locked wheels; no telemetry or model download is required. OS tools are optional detected prerequisites, and portable Blender/FFmpeg can be placed in `<INSTALL_ROOT>/tools`.

## First solve

In the GUI choose **Synthetic Example / Tutorial**, select a parent directory, and use **Solve → Run Full Pipeline…**. Review the solve, inspect **Viewer** and **Quality Control**, then choose **Export → Export Animation…**, select FBX and keep validation enabled. The example supplies measured synthetic observations; it does not pretend to run neural inference. Follow [Your First Multi-Camera Solve](docs/manual/quick-start.md).

The same example runs unattended:

```bash
openmocap demo --output outputs/demo --cameras 8 --frames 36
openmocap gui outputs/demo
openmocap qc outputs/demo
openmocap export outputs/demo --format fbx --output outputs/demo/exports/actor.fbx
```

Representative outputs are in `outputs/demo/outputs/{joints.npz,trajectory.npz,body_pose.npz,animation.npz,contacts.json,cameras.json,qc/report.json,qc/report.html}` and `outputs/demo/exports/{actor.npz,actor.bvh,actor.fbx}`. Generated outputs are ignored by Git.

## Real projects

Use **New Project**, **Cameras → Register media…**, **Import calibration…**, **Synchronization**, and **Actor → Inference checkpoints**. Camera calibration must be measured/imported before a metric reconstruction. New cameras deliberately have no invented calibration. Put licensed body assets in `<INSTALL_ROOT>/models/body`, and user-supplied, compatible ONNX checkpoints in `<INSTALL_ROOT>/models/pose` or `<INSTALL_ROOT>/models/segmentation`. Body-family selection fails if its required assets are absent. No silent fixture fallback exists.

For the selected public detector, 133-landmark pose and segmentation models, run
`python scripts/fetch_models.py --asset all` inside the activated environment,
then use **Actor → Inference checkpoints → Load installed whole-body preset…**
and choose `configs/pose/rtmpose-wholebody-yolox.yaml`. The downloader keeps weights
outside Git and records hashes; the GUI applies their exact preprocessing and
landmark order. See [public model setup](docs/perception/PUBLIC_MODELS.md).
Load `configs/segmentation/pphumanseg.yaml` to add the separate mask settings.
Actual pretrained downloads/inference have not yet been validated: sandbox
network permission stalled before execution. This command does not obtain gated
SMPL-family files, install GPU drivers, or turn the CPU environment into a tested
CUDA environment. The current bootstrap is verified on Linux only.

```bash
openmocap project init /workspace/openmocap-install/projects/take01 --name Take01 --cameras 4
openmocap import-camera /workspace/openmocap-install/projects/take01 CAM_01 /absolute/path/camera01.mov
openmocap import-calibration /workspace/openmocap-install/projects/take01 /absolute/path/cameras.yaml
openmocap run /workspace/openmocap-install/projects/take01
openmocap qc /workspace/openmocap-install/projects/take01
```

Raw observations are preserved. Reversible manual overrides, locks and provenance control effective data; upstream edits mark downstream results stale. Authoritative cameras remain locked. Strong measured joints constrain fitting, contacts and optional kinematic refinement; learned priors apply only to weak evidence. See [configuration](docs/configuration.md), [calibration](docs/calibration/README.md), [body assets](docs/smpl/body-model-contract.md) and [manual](docs/manual/README.md).

## Verification and limitations

```bash
ruff check .
ruff format --check .
pytest
QT_QPA_PLATFORM=offscreen pytest tests -q
./scripts/doctor.sh --json
```

The executed demo has eight asynchronous cameras, 35 output frames, median geometric reprojection 0.548 px and fitted-body median 0.567 px. Its clean FBX re-import verifies 24 bones, 736 weighted vertices, animation, metric root motion and frames 1–35. Synthetic numeric accuracy does not establish accuracy on real imagery. Current limitations include unvalidated real-model/inference assets, incomplete multi-person orchestration and advanced correction UI, no silhouette shape objective, no dynamics simulation, no Alembic, and a USD-enabled Blender backend requirement and incomplete DCC validation.

Code is Apache-2.0. Restricted SMPL-family assets and model checkpoints are not distributed. Qt/Blender/FFmpeg have separate licensing obligations; see [dependencies](docs/DEPENDENCIES.md) and [models](docs/MODEL_LICENSES.md). [Contributing](CONTRIBUTING.md) explains development and geometry governance.
