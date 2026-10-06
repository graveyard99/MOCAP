# Detection and tracking

The detector finds person boxes; local tracking links detections within each camera. Camera geometry belongs in later association and reconstruction.

For real RGB inference, open Actor → Inference checkpoints. Select the detector backend and, for ONNX, its compatible local checkpoint. Apply inference setup. Choose Detection in the navigator and Run selected stage…. The job panel reports processed camera timestamps, warnings and failures.

If an installed inference preset is supplied, click **Load installed whole-body preset…** and select its YAML file under the repository's `configs/` directory. The importer resolves installation-root model paths and checks that checkpoint files exist before changing the project. Missing files and unsupported preset settings produce an actionable error. This imports inference settings only: calibration, camera timing, metric scale and body-model selection retain their existing values. **Edit → Undo** restores the preceding inference settings in one step. Loading a preset is configuration, not proof that inference succeeded; run the stage and inspect its reported status.

The repository includes the [RTMW WholeBody133 + YOLOX HumanArt preset](../../configs/pose/rtmpose-wholebody-yolox.yaml). Its checkpoints are installed separately beneath the installation root's `models/pose/` and `models/detection/` directories. Read its `license_notes` and the [model licensing inventory](../MODEL_LICENSES.md) before using those assets for a production project; the software's license does not grant checkpoint or training-data permissions. A loaded preset configures the detector and pose preprocessing together, so ordinary operators do not need to edit codec configuration.

HOG is a CPU baseline with limited modern capture quality. Its results carry a warning rather than claiming production detector quality. A supplied ONNX detector must match the adapter's supported preprocessing/output contract; arbitrary checkpoint filenames do not guarantee compatibility.

Tracking currently links boxes using temporal IoU and assignment. An appearance helper exists but is not connected to orchestration. Cross-camera actor identity is a separate geometry-weighted stage. Multiple local detections can exist, but full production multi-actor orchestration has not been validated.

The synthetic tutorial contains known observations. Running Detection without a pose checkpoint on that project reports imported observations available and WARNING. No neural detection has occurred, and its status must not be mistaken for detector acceptance.

Raw detections are stored beneath raw/ with model hashes and processing metadata. Re-running later numerical stages does not destructively smooth or rewrite those original records. Detailed person-box/identity editing overlays are still incomplete in the GUI.
