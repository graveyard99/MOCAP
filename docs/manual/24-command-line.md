# Command line

The CLI calls the same service as the desktop. Activate the isolated environment first:

    source /workspace/openmocap-install/openmocap-vfx/scripts/activate.sh
    openmocap doctor
    openmocap gui

Create a project and register measured inputs:

    openmocap project init /workspace/openmocap-install/projects/take01 --name Take01 --cameras 4
    openmocap import-camera /workspace/openmocap-install/projects/take01 CAM_01 /path/to/cam01.mov
    openmocap import-calibration /workspace/openmocap-install/projects/take01 /path/to/cameras.json
    openmocap run /workspace/openmocap-install/projects/take01
    openmocap qc /workspace/openmocap-install/projects/take01
    openmocap export /workspace/openmocap-install/projects/take01 --format fbx --output /path/to/actor.fbx

The new project intentionally lacks invented calibration, metric scale, observations and licensed assets. Supply those measured inputs before running.

For a self-contained demo:

    openmocap demo --output /workspace/openmocap-install/openmocap-vfx/outputs/demo --cameras 8 --frames 36

Add --moving to test imported camera motion. Use --no-fbx when Blender is unavailable; NPZ/BVH still run. The demo command creates generated data and executes actual downstream reconstruction/fitting/QC/export.

Individual commands include ingest, calibrate, sync, detect, pose2d, segment, associate, triangulate, fit-shape, fit-motion, contacts and physics. A selected numerical stage may rerun dependent stages rather than acting as a completely independent executable. Run openmocap --help and command --help for authoritative arguments. --verbose precedes the subcommand and includes detailed failure diagnostics.
