# Installation

The original cloud build was installed at /workspace/openmocap-install/openmocap-vfx, with its Python environment at /workspace/openmocap-install/env. On your desktop choose a writable absolute installation root and clone [graveyard99/MOCAP](https://github.com/graveyard99/MOCAP) into its **openmocap-vfx** child directory. Recreate the environment locally; do not copy the cloud env. Keep models, caches and temporary files beneath the selected root. The repository never needs a global pip installation.

A Linux graphical desktop is needed for ordinary operation. A headless machine can run the CLI and offscreen tests. Supported Python versions are 3.11–3.13; this build was tested with Python 3.12. CPU geometry does not require CUDA. GPU drivers, display libraries and hardware runtimes are system prerequisites; the installer does not silently replace them.

From the repository, bootstrap and activate the isolated environment. The installer resolves INSTALL_ROOT to the repository parent; OPENMOCAP_INSTALL_ROOT can explicitly select it:

    bash scripts/bootstrap.sh
    source scripts/activate.sh
    openmocap doctor

Launch the generated OpenMocap.sh in your installation root, or run:

    openmocap gui

The launcher uses the isolated interpreter automatically. Model files belong beneath your installation root's models directory. Use user-owned licensed numerical NPZ body assets, not arbitrary downloaded pickle files. Compatible ONNX checkpoints are also supplied separately.

Project Health checks Python imports, write access, GPU/PyTorch visibility, FFmpeg and Blender. Blender is needed for FBX export and clean-scene re-import validation. The current machine provides /usr/bin/blender; a supported portable installation can be configured separately. Missing GPU support does not prevent the synthetic solve.

Linux Qt also needs system EGL/OpenGL/xkbcommon runtimes, even for offscreen tests. The first hosted runner passed doctor but failed QtWidgets import because libEGL.so.1 was missing. Doctor's top-level package check does not yet detect that failure. A technician can verify the native Qt import with `python -c "from PySide6.QtWidgets import QApplication"` inside the isolated environment. The CI runner installs the relevant Ubuntu packages explicitly; desktop bootstrap does not make system package changes.

## GPU development and real capture validation

A dedicated workstation is not required. A GPU-enabled cloud environment can
run the same inference and numerical validation. A local graphical desktop or
remote desktop is useful for interactive UI testing and DCC round trips. The
current installation has CPU ONNX Runtime and no visible GPU; CUDA execution has
not been validated. Installing model files alone does not change this.

For planning, an NVIDIA GPU with 12–16 GB VRAM is a practical starting point for
batched inference; 24 GB VRAM, 64 GB system RAM and fast NVMe storage provide more
room for whole-body models and larger captures. These are engineering estimates,
not measured project minimums or a promise of real-time performance. Process
cameras in batches rather than allocating every camera to the GPU simultaneously.
An existing machine should be tested before buying hardware.

The GPU runtime must match the GPU driver and the selected ONNX Runtime CUDA/cuDNN
build. See the official [CUDA execution provider requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).
GPU Python packages and user-space runtimes should remain inside the installation
root; operating-system drivers remain a system prerequisite. NumPy/SciPy geometry
and the current body fitter remain CPU computations. GPU-enabled ONNX inference
does not automatically accelerate those stages.

Real validation also needs a short take of one consenting performer, original
camera media/timestamps, trusted intrinsics/extrinsics, a metric reference and a
measured floor. At least two overlapping views are necessary; four to eight
surrounding views make a more useful initial test of occlusion, body proportions
and contacts. A clap/flash event and a take containing standing, walking, turning
and planted feet help diagnose timing and reconstruction. Synthetic success and
single-image inference checks do not establish real multi-camera accuracy.
