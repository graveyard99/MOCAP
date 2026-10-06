# Technical reference

Canonical world: metres, +Y up, explicit origin and floor. OpenCV cameras: +X right, +Y down, +Z forward; pose rotation/translation maps world coordinates into camera coordinates. Axis/unit conversions belong in geometry.coordinates and the exporter, not ad hoc viewer flips.

Canonical camera records contain id, intrinsics (fx/fy/cx/cy/width/height, distortion/model), pose (rotation/translation), optional trajectory, time_mapping (scale/offset/lock), quality, enabled/locked and source provenance. Trajectory samples interpolate orientation and translation; projection includes the original lens model.

Native 2D records identify camera_id, camera_timestamp, optional world_timestamp, person_id, joint_id, xy and confidence. Root observation indices used by manual corrections refer to the preserved source list. Camera IDs and image dimensions must match calibration.

Key outputs are joints.npz, trajectory.npz, animation.npz, contacts.json, cameras.json and qc/report.json plus report.html. Animation preserves constant shape/topology, rest joints, hierarchy, weights, rotations and root positions. Reprojection diagnostics distinguish measured geometry from the fitted character.

Body NPZ assets require numerical v_template, f, shapedirs, posedirs, J_regressor, weights and parents or kintree_table; optional joint_names maps the skeleton. No object/pickle asset is loaded implicitly. Licensed topology and coefficients are not distributed.

Checkpoints record input/config hashes, code/schema versions and model fingerprints. Export/report provenance includes environment, dependency lock, source hashes and configuration. Exact covariance is not available at every stage: propagated scalar confidence is a quality indicator, not a calibrated universal probability.

See [architecture](../architecture/), [SMPL asset contract](../smpl/body-model-contract.md), and [BUILD_STATUS](../BUILD_STATUS.md) for implementation boundaries.

For GUI calibration import into a new project, include a world wrapper explicitly confirming units: metres, up: Y, metric_scale: 1, origin and calibrated floor. Pipeline outputs/cameras.json is an example of this format. Camera-only files do not supply an unknown project-world scale automatically.
