# Executed validation

Validated in the managed Linux x86-64 environment using project-local Python 3.12.14.
The project is an engineering preview. Synthetic fixtures do not validate the accuracy
of real pretrained models, footage, licensed SMPL assets or external DCC imports.

## Desktop handoff revalidation: 2026-10-06

- `ruff check .` and `ruff format --check .`: passed; 89 Python files formatted.
- `QT_QPA_PLATFORM=offscreen pytest --basetemp="$OPENMOCAP_INSTALL_ROOT/temp/pytest-handoff" -q`: **147 passed in 36.96 seconds**, no skips.
- `uv build --offline --out-dir "$OPENMOCAP_INSTALL_ROOT/downloads/dist" .`: wheel and source distribution built from cached backend dependencies.
- Re-executed eight-camera/36-input-frame demo: 35 output frames, geometric median 0.548006 px and fitted-body median 0.567471 px. NPZ/BVH and actual rigged/skinned FBX exported; clean Blender re-import maximum joint/skin error 1.77843e-6/2.14441e-6 m, 24 bones and 736 weighted vertices.
- Doctor: healthy core, isolated Python, FFmpeg/Blender detected; no actual model assets, GPU or PyTorch. Healthy does not mean those external components are ready.
- `bash -n` passed for bootstrap/activation/doctor/launch; fresh isolated `python -I` imported the package and major dependencies.
- Independent source audits identified unfixed numerical, UI and export-state defects outside current test coverage; see [PROJECT_STATE.md](../PROJECT_STATE.md). This documentation handoff does not change application behavior.
- Final source hygiene/link audit examined 178 tracked/new legitimate files and 64 Markdown files: no file over 5 MiB, forbidden runtime/model/media assets or credential-pattern matches; all local Markdown targets resolve. Required source/manual images are tracked, generated/private ignore guards passed, and `git diff --check` passed. Pattern scanning is a bounded check, not a proof that arbitrary credentials cannot exist.

Earlier results below are retained as historical evidence, not substituted for the fresh handoff checks.

## GitHub publication and first hosted run

Published all 178 source/handoff files to [graveyard99/MOCAP](https://github.com/graveyard99/MOCAP) on `main`. Initial publication commit `8193f3b37875075407c3ef08a7ceca5936541abe` has tree `a9cfdb74fa17fea9ad2bc88c71acfe471bf0f37a`, exactly matching local snapshot `012bff214889c8406acc79768fb0a243b3ec4e80`, including file contents and executable modes. GitHub API publication was used because the cloud Git proxy was unreachable; the original cloud history is retained in the transfer bundle.

The first [hosted run](https://github.com/graveyard99/MOCAP/actions/runs/37510168321) used Python 3.12.15 and successfully installed all 31 locked distributions, ran doctor, and passed lint/format. Test collection failed with `ImportError: libEGL.so.1`; the runner had no FFmpeg/Blender either. The workflow now installs `libegl1`, `libopengl0` and `libxkbcommon0` on the disposable Ubuntu runner. Bootstrap remains isolated and does not make that system change on desktops. Subsequent Actions results must be inspected before claiming hosted tests passed.

## Initial build checks

- `source scripts/activate.sh`
- `ruff check .`: passed.
- `ruff format --check .`: all 83 Python files formatted.
- `QT_QPA_PLATFORM=offscreen pytest --basetemp="$OPENMOCAP_INSTALL_ROOT/temp/pytest-final-threads" -q`: **127 passed in 32.34 seconds**, zero skipped.
- `scripts/bootstrap.sh`: idempotent; all 31 installed packages checked against `uv.lock`.
- `scripts/doctor.sh --json`: healthy; all core imports, isolated prefix and write permissions confirmed.
- Fresh-shell imports, shell syntax, actual GUI launcher event-loop smoke, manual links and license inventory: passed.
- Source audit: no file over 5 MiB, no checkpoints/licensed assets/footage/export binaries, and no credential-pattern matches among 162 inspected source files.

The suite includes noiseless 2/3/8/30-camera reconstruction; noise, outliers, missing
observations, degenerate baselines; distortion/coordinate round trips; moving
camera interpolation; metric displacement; affine offset/drift and bounded
reprojection synchronization; native asynchronous trajectories; persistent shape;
learned-prior precedence; locked calibration; planted versus swing contacts;
checkpoint validity; raw override identity; real FFmpeg VFR decoding; actual ONNX
contract fixtures; Qt job/project/QC/correction workflows; actual skinned FBX
re-import, metric joints, deformed vertices and pose correctives.

A repeat GUI test initially exceeded its 40-second timeout during concurrent
numerical jobs. Profiling a controlled one-thread 3-camera/12-frame full solve gave
4.29 seconds. Launch/activation now defaults `OPENMOCAP_NUM_THREADS=1`, with BLAS,
OpenMP and MKL limits; users can explicitly increase it after profiling their take.
The GUI test has safe worker cleanup and a 120-second deadline. Final checks passed.

## Reproducible demonstrations

    openmocap demo --output outputs/demo --cameras 8 --frames 36
    openmocap demo --output outputs/moving-demo --cameras 5 --frames 24 --moving --no-fbx

The principal example produced 35 output frames from eight asynchronous calibrated
cameras and 6,358 native observations. 6,319 observations contributed; 39 were
rejected/flagged. Median original-plate geometric error was 0.548 px and median
fitted-body error 0.567 px. Deliberate pixel outliers remain in the full residual
statistics, rather than disappearing from QC.

FBX re-import verified 24 bones, 736 vertices with skinning weights, animated local
rotations and metric root motion, frames 1–35, metres and documented axes. Maximum
joint round-trip error was 9.70e-7 m; maximum deformed-vertex error was 1.10e-6 m.
NPZ and BVH were also exported and validated. Outputs are under `outputs/demo/` and
are ignored by Git. QC records the generating Git revision, code/schema version,
configuration, dependency-lock hash and observation hash.

The moving-camera example completed with five timestamped pose trajectories and
median reprojection 0.543 px. A camera that covers only a short interval no longer
trims the full take supported by other cameras. Native-time ray fitting rejects
insufficient-view/ill-conditioned bundles and preserves strong initial geometry.

## External limits actually checked

- No GPU/CUDA/ROCm device was visible. CPU geometry and ONNX CPU execution worked.
- PyTorch is absent and is not required by the implemented CPU/ONNX pipeline.
- No actual licensed SMPL-family asset or production pose/segmentation checkpoint
  was supplied. Numerical body-model and neural-contract fixtures are explicit.
- System FFmpeg 7.1.5 and Blender 4.3.2 were detected and used; neither was modified.
- USD export was attempted and failed because this Blender build has no USD export
  operator. No successful USD character is claimed. A USD-enabled backend and
  skeletal round-trip validation remain required.
- No Maya/Houdini/Cinema 4D/Unreal import, hosted GitHub CI or real production capture
  validation was performed.

See [BUILD_STATUS](BUILD_STATUS.md) for feature-by-feature incomplete work.

## Follow-up model preparation and desktop validation

Executed after adding public model acquisition, inference presets, segmented
doctor diagnostics and GUI preset import:

- `ruff check .` and `ruff format --check .`: passed, 89 Python files formatted.
- `QT_QPA_PLATFORM=offscreen pytest --basetemp="$OPENMOCAP_INSTALL_ROOT/temp/pytest-assets-followup" -q`:
  **145 passed in 32.42 seconds**.
- The suite includes full 133-landmark preset import/apply/undo, whitelist archive
  extraction, explicit MMDeploy/raw-YOLOX codecs, PPHumanSeg preprocessing and
  score-plane argmax, and stale raw-stage status after inference overrides.
- After source freeze, the final run with `--basetemp="$OPENMOCAP_INSTALL_ROOT/temp/pytest-assets-final"`
  passed **147 tests in 32.35 seconds**. Additional tests verify the direct
  segmentation publisher checksum and failure of an advertised CUDA provider
  that initializes only CPU. Ruff lint/format checks passed on 89 Python files.
- Re-ran `openmocap demo --output outputs/demo --cameras 8 --frames 36` after
  the source commit `5dabb9b`: 35 frames, geometric median 0.548 px, validated
  NPZ/BVH/rigged FBX. Clean FBX re-import joint error was 1.78e-6 m and skin-vertex
  error was 2.14e-6 m. The deterministic input reproduces the same QC statistics;
  small floating-point export differences remain far below the test tolerance.
- `python scripts/fetch_models.py --asset pose --timeout 5` failed immediately
  with curl exit 7 because the sandbox could not connect to the download proxy.
  Its partial file was removed. Network-enabled attempts stalled before command
  execution awaiting sandbox permission; they were aborted. These results do
  not establish a failure of the official model host.
- No actual pretrained checkpoint, licensed body asset or CUDA inference is
  counted as validated. The complete GPU installer remains unimplemented; the
  verified dependency bootstrap targets Linux and CPU ONNX Runtime.

The source can be archived without environments, checkpoints, capture media or
generated results for desktop validation. Desktop OS, GPU/VRAM, driver, available
execution connection and runtime compatibility must be established before
claiming GPU support. Model-file presence in doctor output is labelled
`file_presence_only`, and remains distinct from executed inference validation.
