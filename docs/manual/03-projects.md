# Projects

Choose New Project on the home screen. The three-page wizard collects project name/directory, optional footage folder, camera count/prefix/FPS, calibration source/file, metric source/floor, timing method, actor/model assets and delivery FPS/output folder.

1. Choose an empty project directory. Camera counts from two through 256 can be entered.
2. Enter stable camera names. The prefix generates numbered IDs; these IDs must match imported calibration and observation records.
3. Select the calibration and timing sources actually available. Selecting a source records intent; it does not invent measurements.
4. Choose smpl, smplh or smplx with its licensed NPZ file for real use. Choose fixture only for the synthetic tutorial.
5. Finish, then register each camera's media and import measured calibration.

The project stores project.yaml, media references, calibration, raw observations, caches, overrides, outputs and exports. Source plates may remain outside the project; registration keeps their paths. Keep those references valid when archiving.

Open Project accepts a project configuration file. Recent entries reopen the project directory. Project → Save Project persists settings. Duplicate Project creates a new destination through the shared service and excludes expensive caches/outputs; source media references may still point to the original files.

Project Settings edits project name and output FPS. Returning to Home or closing the application does not delete a solve. Processing must finish or be canceled before changing projects.
