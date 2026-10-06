# Build status

This is an implemented and tested engineering preview, not a production-ready release.
The reproducible synthetic pipeline has executed through a validated rigged/skinned FBX.
No licensed SMPL assets or production inference checkpoints were supplied.

| Feature | Status | Evidence / limits |
|---|---|---|
| Isolated environment, deterministic lock, CLI | WORKING | Project-local Python 3.12, uv.lock, doctor healthy |
| Camera projection, distortion, metric coordinates | WORKING | Synthetic camera and round-trip tests |
| Robust arbitrary-N reconstruction | WORKING | 2/3/8/30-view tests; outliers/missing/baseline checks |
| Continuous asynchronous reconstruction | WORKING | Native-time spline ray solve, unseen-time acceptance |
| Camera calibration/moving-camera integration | PARTIAL | Target/PnP/import/refinement APIs; imported moving trajectories; GUI correspondence editing incomplete |
| Native timestamp/media ingestion | WORKING | FFprobe native timestamps, dropped/duplicate handling and bounded media decoding tests |
| Affine synchronization | WORKING | Signal/event correspondence and bounded reprojection APIs; manual GUI clocks |
| RGB automatic inference | BLOCKED_EXTERNAL | Local HOG/ONNX integration and explicit schema adapters implemented; user-supplied compatible production checkpoints/real-capture validation required |
| Detector/pose/segmentation backend integration | PARTIAL | HOG and explicit local ONNX contracts tested; no production whole-body/mask checkpoints validated |
| Public model acquisition and GUI preset setup | PARTIAL | Official-source downloader, isolated manifests, archive-safety tests and full 133-landmark GUI preset implemented; actual transfers and pretrained inference blocked by sandbox network permission, not model-host credentials |
| Cross-camera identity | PARTIAL | Geometry interfaces; full multi-actor orchestration not validated |
| Persistent synthetic rig and shape | WORKING | One rest mesh/shape, per-frame local rotations and metric root |
| SMPL/SMPL-H/SMPL-X asset integration | PARTIAL | Numerical LBS and contract fixture tests; actual licensed assets required |
| Actual licensed SMPL capture validation | BLOCKED_EXTERNAL | User must obtain licensed numerical model assets |
| Silhouette-based shape fitting | NOT_STARTED | No silhouette objective claimed |
| Contacts | WORKING | Four floor-aware contact states, bounded corrections, tests |
| Physics | PARTIAL | Optional bounded kinematic refinement; no dynamics simulation |
| Dynamics simulation | NOT_STARTED | No force/inertia/balance simulator claimed |
| Scene constraints | PARTIAL | Explicit calibrated plane/scene data contract; stairs/walls/collision-mesh solving absent |
| QC | WORKING | Original-plate geometric and fitted-body JSON/HTML residuals |
| FBX, NPZ, BVH | WORKING | Demo FBX clean Blender re-import, mesh/skeleton/weights/animation/root/range validated |
| USD | BLOCKED_EXTERNAL | Installed Blender lacks USD export operator; compatible USD-enabled backend and skeletal validation required |
| Alembic | NOT_STARTED | A mesh cache is not a rigged character export |
| Retargeting | PARTIAL | Explicit joint naming/mapping; target rest-rig calibration and turnkey DCC adapters absent |
| Manual overrides and checkpoint validity | WORKING | Audited raw+override state, undo/reopen, stale descendants and raw-input hash binding tested |
| Desktop production workflow | PARTIAL | Cohesive Qt GUI, async real solve/export tests and screenshots; advanced correction/calibration tools incomplete |
| Isolated bootstrap, activation, GUI launchers | WORKING | Idempotent frozen bootstrap, fresh-shell imports/cache-root assertions and actual offscreen GUI launcher event-loop smoke passed |
| Operator manual for implemented preview | WORKING | 26 chapters, tutorial and seven actual screenshots; UI/selected-stage walkthrough and relative-link audit completed; unavailable tools explicitly documented |
| CI/pre-commit | PARTIAL | Workflow/hooks authored; matching local Ruff/test commands pass, hosted GitHub execution not performed |

Latest demonstrated result: eight asynchronous cameras, 35 output frames, 6,358 native observations;
6,319 used, 39 rejected/flagged; geometric median reprojection 0.548 px, fitted-body median 0.567 px.
FBX re-import validated 24 bones, 736 weighted vertices, frames 1–35, animated metric root,
and maximum joint round-trip error below 2e-6 m. Deliberately injected outliers remain visible in QC.
**Latest integrated verification: 147 tests passed in 36.96 seconds; Ruff lint and formatting passed.** See [executed validation](VALIDATION.md).

Installation verification: `scripts/bootstrap.sh` was rerun successfully with all 31 installed packages checked. `bash -n` passed for four shell scripts; fresh-shell imports and all project cache-root assertions passed. The root-local GUI launcher entered the Qt event loop under offscreen smoke validation before controlled termination. Local installed-wheel metadata/license notices were inventoried for 31 distributions. See `docs/INSTALLATION_DECISIONS.md` and `docs/licenses/installed-review.json`.

Follow-up asset preparation (2026-10-06): public RTMW-l/YOLOX/PPHumanSeg acquisition and
verification scripts and inference presets have been added. Network-enabled
commands stalled before execution awaiting sandbox permission and were aborted;
no model download or real pretrained-model inference succeeded. The environment
remains CPU-only. A GPU desktop is available to the owner but its OS, hardware,
runtime and remote execution access have not yet been established. The tested
bootstrap remains Linux/CPU; a tested complete GPU application installer is not
claimed. SMPL/SMPL-X assets still require the owner's official portal account.

Handoff source audit (2026-10-06): [PROJECT_STATE.md](../PROJECT_STATE.md) records
confirmed unfixed defects, including singular covariance, sparse shape evidence,
downstream moving-camera confidence, GUI native-PTS/schema/floor/range handling
and export-time stale-input detection. Passing fixtures do not cover these cases.
The demo, doctor, isolated imports, shell checks and wheel/sdist build were rerun.
See [AGENTS.md](../AGENTS.md) and [SESSION_HANDOFF.md](../SESSION_HANDOFF.md) before
resuming development on the desktop. This handoff changes documentation/ignore
rules only; application behavior remains unchanged.
