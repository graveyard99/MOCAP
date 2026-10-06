# ADR 004: shared desktop and automation services

Status: accepted.

Qt is isolated from the numerical modules. Both desktop jobs and CLI commands call
`ProjectService`, which owns project persistence, validation, stage status,
content-addressed checkpoints, manual override precedence and export dispatch.
Worker threads keep long numerical jobs away from Qt's event thread. Cancellation
is cooperative at reconstruction iterations and stage boundaries, and closing an
active job waits for a safe boundary. Completed checkpoint manifests are atomic.

Camera transforms are read-only during body reconstruction. Manual overrides sit
above imported calibration and can be restored to their baseline. Observation
corrections retain raw indices, original input hashes and separate override files;
regenerating raw data refuses to reuse corrections against shifted indices.
Stages display STALE after upstream edits. Export refuses stale upstream dependencies
while permitting a previously exported take to be regenerated after a fresh solve.

The 3D viewport uses Qt raster projection and painter rendering. This works on CPU
and offscreen test machines, with orbit/pan/zoom, skeleton, mesh, frustums, floor,
contacts and confidence. It is an engineering preview: dense scene rendering,
identity corrections, target correspondence editing and silhouette overlays remain
incomplete. A future accelerated renderer can replace the viewer without changing
solver services.

Model inference uses locally supplied ONNX files. GUI controls specify the actual
backend, checkpoint path and joint schema; the synthetic tutorial uses explicitly
imported observations. Available backend adapters never prove a pretrained model's
accuracy. Actual user models and licensed SMPL assets need separate validation.
