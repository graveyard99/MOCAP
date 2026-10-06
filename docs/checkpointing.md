# Checkpoints and reproducibility

Pipeline checkpoints contain schema/code version, content key, output paths and output SHA-256 hashes. A key incorporates configuration, relevant inputs, all package source files and the dependency lock. Source/config/input/lock changes invalidate compatible stage reuse. Output tampering fails hash checks. Stage manifests are atomically replaced only after completion; partial files do not earn COMPLETE status.

Projects retain calibrated cameras and affine clocks, immutable timestamped observations, manual override audit data, reconstructed joints, continuous trajectory, persistent actor rig/body pose, final animation, contacts, QC and export metadata. `body_pose.npz` keeps the pre-contact rig checkpoint while `animation.npz` holds the selected final rig motion. Raw neural stages have their own versioned artifacts so later numerical changes do not require re-running compatible inference.

Manual edits mark downstream results STALE. Observation-index overrides additionally bind the raw-observation input hash; a detector rerun that changes raw records cannot accidentally apply an old disabled index to a different measurement. Restore/reapply explicitly. Closing/reopening retains completed artifacts, overrides and stage state. Cancellation is checked at safe boundaries; an incomplete stage is not reused as valid.

Full solve provenance records configuration, code/schema version, Git commit where available, Python/dependency/device information, lock hash, seed and input/model hashes where applicable. Keep project YAML, checkpoint manifests, QC JSON and export sidecars with a deliverable. Numeric NPZ uses safe arrays/Unicode metadata; untrusted pickle is never required for cached solve data.
