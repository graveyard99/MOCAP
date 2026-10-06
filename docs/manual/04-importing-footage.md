# Importing footage

Open Cameras and select the camera row that should own the plate. Click Register media…. Choose Video / Image File or Image Sequence Folder in the source chooser, then select the source. Repeat for each enabled camera. The synthetic project already registers its generated sequences.

Registration extracts a media index and preserves timestamps where available. Video processing uses presentation timestamps rather than matching equal frame numbers across cameras. An image sequence without capture timing needs a declared nominal FPS; that timing is an assumption, not recovered hardware synchronization.

Inspect Source, Resolution, FPS and Timestamps in the camera table. Calibration dimensions must agree with actual plate dimensions. A resized image with unchanged focal lengths is a different camera model and fails validation or produces misleading residuals.

The plate viewer decodes requested images asynchronously and limits outstanding work while scrubbing. It reads image sequences lazily and uses available timestamp manifests to select the nearest plate in each camera's own clock.

Prefer complete original sequences and reliable timestamp sidecars. Duplicate or dropped frames should remain in the timing record. Do not renumber footage merely to make equal frame numbers appear synchronized.

Ingest can be rerun from the pipeline navigator. Neural stage outputs are cached separately, so changing a later solver setting need not decode and infer the whole take again.
