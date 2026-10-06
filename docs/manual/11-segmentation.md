# Segmentation

Select Segmentation in the navigator and Run selected stage… to process registered media. The stage preserves per-person masks and confidence in raw/masks and raw/silhouettes.json.

Without a semantic checkpoint, the available fallback is rectangle-conditioned OpenCV GrabCut. It separates foreground colour and is not a learned human-semantic model. Its intentionally low confidence and WARNING status reflect that limitation. Clothing similar to the background can produce poor masks.

A compatible ONNX segmentation backend is available through technical perception configuration. The GUI does not yet offer a dedicated segmentation-model picker or mask-editing workspace. Use a technician-managed project configuration when enabling that backend.

Original masks are preserved independently of body fitting. Silhouette-based actor-shape optimization is not implemented in this release. Running segmentation does not cause shoulders, waist or limb thickness to fit those masks.

Detected/projected silhouette overlays are also incomplete in the desktop viewer. Review exported mask files with an image viewer and consult their provenance/confidence before downstream use.

For the synthetic tutorial, segmentation is unnecessary. Leaving it NOT RUN does not invalidate measured joint reconstruction or rigged character export. It would be incorrect to mark this stage COMPLETE merely because an interface or mask directory exists.
