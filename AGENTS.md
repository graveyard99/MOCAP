# OpenMocap VFX: instructions for implementation agents

## Start here

Read [PROJECT_STATE.md](PROJECT_STATE.md) and [SESSION_HANDOFF.md](SESSION_HANDOFF.md), then [README.md](README.md), [BUILD_STATUS.md](docs/BUILD_STATUS.md) and [VALIDATION.md](docs/VALIDATION.md). These records distinguish working code, incomplete features and actual execution evidence. Inspect the current files and Git state before changing anything; the conversation history is not a substitute for the repository.

This is a geometry-first, metric, asynchronous multi-camera mocap **engineering preview**, not a production-ready release. The tested vertical slice uses timestamped synthetic observations and an explicitly generated technical mannequin through a genuinely rigged/skinned FBX. Do not relabel that fixture as SMPL or describe interface/fixture tests as pretrained-model or real-capture validation.

The initial handoff audit found a clean source tree at `56844ca` and no Git remote; the handoff was then committed at `56fb363`. The owner subsequently authorized publication to **https://github.com/graveyard99/MOCAP.git**, now configured as `origin`. Clone into an explicit `<INSTALL_ROOT>/openmocap-vfx` directory, because Git's default `MOCAP` name disagrees with launcher assumptions. The cloud Git proxy is unreachable, so publication uses the authenticated GitHub connector and creates GitHub commits with different IDs from the original cloud development history. The cloud Git bundle preserves that original history; verify the remote source/tree and actual branch head rather than assuming identical commit IDs. Do not reset, discard or overwrite user changes. Check `git status --short`, `git log -1` and `git remote -v` before work. Never inspect credential values to investigate authentication.

## Non-negotiable engineering rules

- Trust measured geometry first: surveyed/calibrated physical geometry, robust multiview evidence, temporal observations, body constraints, biomechanical constraints, learned priors, monocular estimates. Never silently replace authoritative cameras with a learned estimator.
- Preserve source, ownership, confidence and `LOCKED`/`BOUNDED`/`FREE` control states. Camera/body/timing optimizers must respect locks, explicit bounds and diagnostic deltas. Manual changes must be deliberate and auditable.
- Use arbitrary camera lists and whichever views are valid. Reconstruction uses all surviving views; hypothesis pairs are an initializer, not a reason to discard N-view evidence. Require at least two usable independent metric views and report poor conditioning.
- World coordinates are metres, +Y up, with an explicit origin and calibrated ground. Camera transforms are OpenCV world→camera: +X right, +Y down, +Z forward. Centralize conversion in `geometry/coordinates.py`; do not scatter unexplained axis flips or invent metric scale.
- Cameras map local seconds by `t_world = scale * t_camera + offset`. Use native timestamps and continuous trajectories, not matching integer frame indices or nominal FPS. Preserve dropped/duplicate/VFR diagnostics.
- Raw detector observations/masks are immutable. Effective results combine raw data and reversible overrides. Keep raw-input hash binding for index-based corrections, invalidate downstream artifacts and refuse stale export. Do not mark unavailable/skipped stages complete.
- Fit one persistent actor shape/rest rig per take. Later contact/kinematic stages must hold shape/topology/weights fixed. Strong multiview joints constrain the fit; subordinate learned initialization may fill only weak evidence. Default strong-evidence displacement limits are enforced, not silently weakened to make a bad rig fit pass.
- Numerical modules must not import UI. GUI and CLI delegate to the same application/perception services. Long jobs run asynchronously with progress, actionable errors and safe cancellation/checkpoints.
- Preserve separate geometric and fitted-body reprojection QC. A smooth animation, plausible body or low geometric error does not prove the rig agrees with plates. Confidence is an engineering quality signal, not certified metrological accuracy.
- Never claim successful download, CUDA execution, SMPL fitting, export validation or tests without actual evidence. Report `WORKING`, `PARTIAL`, `BLOCKED_EXTERNAL` or `NOT_STARTED` accurately in build status and documentation.
- Honor the user's existing authorization and continue ordinary reversible engineering work autonomously. Ask only for genuinely missing access, hardware information or owner actions; do not repeatedly reconfirm already-authorized personal-use model acquisition.

## Architecture and file ownership map

| Path | Actual responsibility |
|---|---|
| `src/openmocap/types.py` | Typed cameras, clocks, observations, rig/contact/scene/QC contracts, provenance and parameter controls. |
| `src/openmocap/application.py` | `ProjectService`, project lifecycle/validation, overrides, stage states, presets, solve/export orchestration, doctor and reproducibility. |
| `src/openmocap/cli/` | Argparse automation commands; logging/progress/error reporting over the same service. |
| `src/openmocap/ui/` | Qt project manager/wizard, coherent workspaces, camera/actor/export dialogs, viewers, saved preferences/undo and worker jobs. |
| `src/openmocap/perception.py` | Native RGB stage orchestration, provider/model configuration, raw artifacts, explicit schema adaptation/association and inference cache keys. |
| `src/openmocap/io/` | Native media timestamps/bounded decoding and observation serialization/non-destructive overrides. |
| `src/openmocap/cameras/` | Camera projection/undistortion, camera pose/trajectory sampling and helpers. |
| `src/openmocap/calibration/` | OpenCV/target/PnP/import pathways, controlled camera refinement and static-scene COLMAP integration. |
| `src/openmocap/geometry/` | Coordinate conversions, similarity alignment and measured scale helpers. |
| `src/openmocap/sync/` | Affine clocks, event/signal alignment and bounded reprojection timing refinement with a locked gauge. |
| `src/openmocap/detection/`, `tracking/`, `pose2d/`, `segmentation/` | HOG/GrabCut baselines, local ONNX contracts, tracking/search-region feedback, named pose schemas and masks. |
| `src/openmocap/association/` | Geometry-led cross-camera identity evidence and explicit ambiguity; full production multi-actor orchestration remains partial. |
| `src/openmocap/triangulation/` | Robust weighted N-view initialization/refinement, outlier rejection, geometry/confidence/covariance diagnostics. |
| `src/openmocap/trajectories/`, `pipeline/reconstruction.py` | Continuous native-time ray/spline reconstruction and output-time sampling. |
| `src/openmocap/pipeline/cache.py` | Content keys, code/schema/lock/input hashing, atomic manifests and output-integrity checks. |
| `src/openmocap/body_models/`, `fitting/` | Independent numerical SMPL-family coefficient/LBS adapter, safe asset conversion, persistent shape and local-rotation/root-motion fitting. |
| `src/openmocap/contacts/`, `physics/` | Four floor-aware foot channels and optional confidence-bounded kinematic refinement. There is no dynamics simulator. |
| `src/openmocap/qc/`, `retargeting/` | JSON/HTML residual/provenance reporting and explicit joint-name mapping. |
| `src/openmocap/export/` | NPZ/BVH, Blender rig/skin/animation export, clean FBX re-import validation and FPS/range resampling. |
| `src/openmocap/synthetic/` | Deterministic calibrated static/moving camera fixtures with noise, missing data, outliers and asynchronous clocks. |
| `tests/unit`, `synthetic`, `integration`, `end_to_end` | Numerical/contract, acceptance, desktop/export and full pipeline tests. |
| `scripts/`, `configs/`, `docs/`, `.github/` | Isolated setup/launch/acquisition, human-readable examples/presets, operator/engineering documentation and CI/templates. |

Important boundaries: `ProjectService.run(stage=None)` is not a general DAG scheduler that automatically executes every navigator label. It can consume imported keypoints, run configured pose inference, reconstruct/fit/contact/QC, and leave unrelated stages `NOT RUN` or `WARNING`. Segmentation is an explicit stage and currently has no silhouette-fitting objective. Calibration-stage execution validates an imported solution; it does not perform every target/SfM tool automatically. Shape/motion share the current `fit_actor` path. Do not promise separate fully implemented optimizers merely because both stage labels exist.

`PRESETS` in `application.py` define triangulation `max_hypotheses` and `max_iterations`: Preview 16/40, Standard 32/80, High Quality 64/100, Maximum / Final 128/200. Body fitting has the separate `solver.body_max_nfev` setting, default 35 in the service. Do not describe preset iterations as body-fit iterations.

Disabled cameras are excluded from validation/reconstruction; at least two distinct enabled calibrated cameras remain necessary. Export refuses stale dependent stages, resamples local rotations/root motion to requested FPS/range and delivers camera metadata as a JSON sidecar. DCC target labels do not prove tested native import behavior or a distinct conversion per target.

## Installation root and desktop migration

Historical cloud paths were `/workspace/openmocap-install/openmocap-vfx` and `/workspace/openmocap-install/env`. On another machine select an absolute writable installation root and keep the repository at `<INSTALL_ROOT>/openmocap-vfx`. Bootstrap resolves the repository's parent unless `OPENMOCAP_INSTALL_ROOT` is explicitly supplied; keep that override consistent with the repository layout because the generated launcher expects this child name. Print resolved paths before changing the machine.

```text
<INSTALL_ROOT>/
  openmocap-vfx/     source repository
  env/              isolated Python environment
  tools/            local uv and portable executables where available
  models/           body, pose, detection, segmentation assets/manifests
  cache/            package, neural-library, GUI and other caches
  downloads/        archives and build artifacts
  temp/             temporary files and pytest bases
  projects/         user projects and managed working data
```

The verified target is Linux x86-64, Python 3.12.14, CPU geometry and CPU ONNX Runtime. Package metadata supports Python 3.11–3.13. The lock pins CPU `onnxruntime==1.22.0`; PyTorch and CUDA runtime are not installed. GPU desktop OS, GPU/VRAM, driver and access were not established. Bash/Linux launchers are implemented; a tested native Windows/macOS installer or full GPU installer is not. Inspect the actual desktop before making those decisions. Never infer NVIDIA/Linux/administrator access from the existence of a GPU desktop.

Do not copy the cloud `env` or executable Python links to a desktop. Recreate the environment locally from the lock. Existing OS Python, display runtime, GPU drivers, FFmpeg/ffprobe and Blender are external prerequisites; bootstrap does not install or modify them. It installs project Python packages under `env`, copies/acquires uv under `tools`, creates local launchers and runs doctor. There is no hidden model download in bootstrap.

Source `scripts/activate.sh` before developer commands. It sets `UV_PROJECT_ENVIRONMENT`, PIP/UV/Hugging Face/Torch/XDG/Matplotlib/CUDA/Numba cache locations, temporary directories, local tool paths and BLAS/OpenMP/MKL thread limits. `ProjectService.configure_isolation` also sets its applicable caches under the installation root. Preserve isolation: no global pip, no home-directory checkpoints/caches, no project secrets in Git, and never repurpose `HOME` or `CODEX_HOME`. Default numerical threads are one to avoid previously measured oversubscription; profile before increasing `OPENMOCAP_NUM_THREADS`.

The managed cloud's proxy/permission failure is not evidence that publisher hosts are down. Repeated network-enabled calls stalled before execution and were aborted; a bounded default-sandbox curl also failed to connect to the proxy. Do not replay that permission loop. Reassess the new execution environment/network policy, use bounded official-source acquisition when authorized, and report actual failures. Do not copy cloud credentials/proxy/certificate paths into desktop setup or scan secret files.

## Exact Linux commands

Run from the repository; replace the installation location with the actual desktop path. Bootstrap has no positional install-root argument.

```bash
cd /absolute/installation/root/openmocap-vfx
./scripts/bootstrap.sh
source scripts/activate.sh
./scripts/doctor.sh --json
"$OPENMOCAP_INSTALL_ROOT/OpenMocap.sh"
```

The launcher activates the isolated environment automatically. Developer/review commands:

```bash
source scripts/activate.sh
openmocap doctor --json
openmocap gui
openmocap demo --output outputs/demo --cameras 8 --frames 36
openmocap gui outputs/demo
openmocap project init "$OPENMOCAP_INSTALL_ROOT/projects/take01" --name Take01 --cameras 4
openmocap import-camera "$OPENMOCAP_INSTALL_ROOT/projects/take01" CAM_01 /absolute/path/camera01.mov
openmocap import-calibration "$OPENMOCAP_INSTALL_ROOT/projects/take01" /absolute/path/cameras.yaml
openmocap run "$OPENMOCAP_INSTALL_ROOT/projects/take01"
openmocap qc outputs/demo
openmocap export outputs/demo --format fbx --output outputs/demo/exports/actor.fbx
```

New projects deliberately lack calibrated geometry/metric scale; importing one source and invoking `run` is not sufficient until all required inputs exist. Individual stages take the same project argument: `ingest`, `calibrate`, `sync`, `detect`, `pose2d`, `segment`, `associate`, `triangulate`, `fit-shape`, `fit-motion`, `contacts`, `physics`, `qc`. Use `openmocap --help`/subcommand help for real options. CLI debug logging uses the global option before the command:

```bash
openmocap --verbose run outputs/demo
```

CLI progress is emitted on stderr, result/doctor data on stdout; verbose errors include tracebacks. GUI worker exceptions include traceback text and job status. Inspect saved project state, `cache/checkpoints`, raw observation/mask provenance and `outputs/qc/report.json`/`report.html` before modifying solver weights. Doctor's healthy status checks core imports/repository/isolation; `model_status.validation=file_presence_only` does not validate the model, CUDA initialization or an export backend's capabilities.

## Build, lint and tests

```bash
source scripts/activate.sh
ruff check .
ruff format --check .
QT_QPA_PLATFORM=offscreen pytest --basetemp="$OPENMOCAP_INSTALL_ROOT/temp/pytest-desktop" -q
bash -n scripts/bootstrap.sh scripts/activate.sh scripts/doctor.sh scripts/launch.sh
```

Focused commands, when relevant:

```bash
pytest tests/synthetic/test_geometry_acceptance.py tests/unit/test_camera_models.py -q
pytest tests/unit/test_body_fitting.py tests/integration/test_animation_export.py -q
pytest tests/unit/test_perception.py tests/unit/test_pphumanseg.py tests/unit/test_public_model_contracts.py -q
QT_QPA_PLATFORM=offscreen pytest tests/integration/test_ui.py -q
```

FBX tests require Blender; actual video tests require FFmpeg/ffprobe. Test skips on a new desktop mean those checks were not executed there. Offscreen Qt is a test setting, not normal interactive display operation. The fresh handoff validation passed 147 tests in 36.96 seconds, with no skips in this cloud installation; do not claim that count for a new machine without running it.

The following local uv 0.12.19 build command was executed successfully during handoff and produced both wheel and source distribution:

```bash
source scripts/activate.sh
uv build --offline --out-dir "$OPENMOCAP_INSTALL_ROOT/downloads/dist" .
```

An offline build needs cached build-backend dependencies; the cloud cache had them, but a fresh desktop may not. Report missing cache rather than silently accessing the network. The build backend pins Hatchling 1.27.0. A wheel/sdist is a Python package artifact, not a complete GPU desktop installer, portable interpreter or bundled Qt/Blender/FFmpeg/models application. Keep build output under the installation root. Dependency changes require reviewing `pyproject.toml` and `uv.lock`, licenses and actual runtime compatibility; normal installs use `uv sync --frozen`, not an unreviewed upgrade.

CI is `.github/workflows/ci.yml`: Ubuntu/Python 3.12, isolated bootstrap, Ruff, tests, CPU no-FBX demo and doctor. Its first hosted run passed bootstrap/lint/format but failed Qt test collection because the runner lacked `libEGL.so.1`. The workflow explicitly installs EGL/OpenGL/xkbcommon system runtimes on the disposable CI runner; bootstrap never installs these on the user's machine. The [corrected hosted run](https://github.com/graveyard99/MOCAP/actions/runs/37510599267) passed **143 tests with four skips in 45.96 seconds**, plus lint/format/demo/doctor. Two FBX tests skipped without Blender and two media tests skipped without FFmpeg/ffprobe; cloud validation with those tools passed all 147. Doctor's top-level PySide6 import alone cannot establish QtWidgets native-runtime health; separately verify that import or run UI tests. Optional hooks use `.pre-commit-config.yaml`; installing hooks must retain project-local caches.

## Assets, licensing and current external limits

At handoff there are **zero actual ONNX/body-model checkpoints** under the installation root. A request manifest, YAML preset, generated ONNX test fixture or model file-presence check is not an acquired/validated production checkpoint.

The user has authorized personal noncommercial artistic experimentation and public model acquisition. The reviewed SMPL/SMPL-X grants include artistic projects subject to their full terms, but the owner must register/accept the portal agreement and obtain the numerical files. Do not ask for the owner's password, accept terms silently, scrape credentials or bypass gated access. Put owner-provided assets under `models/body`; use safe numerical NPZ where possible. Legacy pickle loading requires explicit trusted-file conversion and can execute code. Never silently substitute the procedural fixture for a requested SMPL family.

Original project code is Apache-2.0. Dependencies, Qt, Blender/FFmpeg binaries, pretrained checkpoints, training data and body assets have separate terms. Read [DEPENDENCIES.md](docs/DEPENDENCIES.md), [MODEL_LICENSES.md](docs/MODEL_LICENSES.md), [PUBLIC_MODELS.md](docs/perception/PUBLIC_MODELS.md) and [PPHumanSeg provenance](docs/segmentation/public-model.md) before acquisition/distribution. Restricted coefficients/checkpoints/private footage/envs/caches/export binaries/secrets must not enter Git or a public source archive. `.gitignore` is a guard, not proof that every private binary extension is covered; inspect added/tracked files.

When network access is available, existing bounded public acquisition commands are:

```bash
source scripts/activate.sh
python scripts/fetch_models.py --asset all --timeout 40
python scripts/verify_public_models.py /absolute/path/authorized-person-image.jpg --provider CPUExecutionProvider
```

The shared downloader selects RTMW whole-body, YOLOX HumanArt and PPHumanSeg; it stores weights/manifests locally and validates PPHumanSeg's published LFS size/SHA-256. The verification script exercises detector/pose on an authorized real image, not metric accuracy or segmentation quality. CUDA verification must prove the requested provider initialized; CPU fallback is a failure for an advertised CUDA run. Do not install conflicting CPU/GPU ONNX packages casually or claim GPU performance without actual evidence. Perception path tokens `${INSTALL_ROOT}`, `${OPENMOCAP_INSTALL_ROOT}` and `<INSTALL_ROOT>` have an explicit resolver; do not assume arbitrary shell expansion throughout all project fields.

Current gaps remain documented: actual licensed-model and real-pretrained capture validation; silhouette-based body shape; dynamics; advanced correspondence/identity/manual-range UI; complete multi-person orchestration; turnkey retargeting/DCC roundtrips; Alembic; and a tested complete GPU desktop installer. The tested system Blender lacks USD export. Actor height is metadata, not an implemented automatic scale solve.

Read PROJECT_STATE's confirmed defects before modifying solvers or UI. In particular, degenerate pseudoinverse covariance can falsely imply precise depth; sparse shape evidence can be zero-weighted; later camera objectives omit moving-trajectory confidence. The GUI has native-PTS, current-clock, whole-body edge, floor and full-take reset bugs. Export detects recorded STALE states but does not rehash externally edited input/config files: rerun before export after such edits. Stage invalidation and cancellation are incomplete. These are documented source findings, not fixed by the passing synthetic suite.

## Changes and handoff discipline

Use typed contracts, informative exceptions, deterministic seeds where relevant, inspectable residuals and meaningful tests. Keep source/schemas/data conventions explicit; unknown calibration/units/scale/timestamps must fail loudly. Profile before speculative optimization. Avoid dependencies that entangle numerical code with UI or silently introduce model downloads.

For an authorized implementation, make the change concrete, run appropriate checks, repair failures and update the actual operator/manual/status records. For a documentation/migration request, preserve application behavior. Keep PROJECT_STATE and SESSION_HANDOFF current with exact paths, actions, results, blockers and next work. Do not overwrite historical validation with a new unexecuted claim. Publication, communication to others and destructive system changes require their actual user authorization; preparing a clean local repository does not require publishing it.
