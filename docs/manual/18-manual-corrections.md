# Manual corrections and undo

Corrections are non-destructive. Original detector/calibration records remain available; audited user overrides produce the effective result used by later processing.

To disable a bad 2D measurement, click its green point in a plate viewer, or double-click its observation row in Quality Control. The inspector identifies the raw observation. Click Disable measured observation. Restore measured observation reverses that override. Rerun the solve afterward; existing downstream states become STALE.

Use Cameras → Enable / disable for whole-camera exclusion. Edit selected… exposes calibration and timing locks, quality and explicit manual metadata. In Quality Control, Mark range as contact records a foot interval. Actor and model collects model selection and provided measurements.

Edit → Undo / Redo supports these parameter changes in the current application session. Saving/reopening retains the overrides; the undo command stack itself is session-local. Raw result files remain unchanged, and expensive completed solves are not deleted merely by editing a value.

Expert override controls expose service keys and JSON scalar values for technician work. Ordinary camera, actor, timing and contact changes have normal controls. Interactive identity reassignment, per-camera time-range exclusion and general bad-range authoring are not fully implemented.

Edits and project switching are blocked while a processing job is active. Cancel or finish the job before changing its inputs so concurrent UI operations cannot silently rewrite a running solve.

If raw inference output is regenerated or reordered, stored observation-correction fingerprints no longer match. The solve refuses to apply those corrections to different source data. Refresh QC, inspect the new source observations, and restore/reapply corrections deliberately.
