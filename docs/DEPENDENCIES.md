# Dependencies and redistribution

The project's original code is Apache-2.0. Dependencies are installed as separate, unmodified packages using the checked-in `uv.lock`; their licenses do not become Apache-2.0. License review must be repeated before binary redistribution, checkpoint acquisition or changing the lock. No model checkpoint or third-party executable is committed.

| Component | Locked version / role | Upstream license and authoritative source |
|---|---|---|
| NumPy | 2.2.6; arrays/LBS | BSD-3-Clause; https://github.com/numpy/numpy/blob/v2.2.6/LICENSE.txt |
| SciPy | 1.15.3; optimization/splines | BSD-3-Clause; https://github.com/scipy/scipy/blob/v1.15.3/LICENSE.txt |
| OpenCV contrib headless | 4.11.0.86; camera/lens/targets/HOG | MIT for opencv-python wrapper; Apache-2.0 for OpenCV; wheel bundled libraries have separate notices; https://github.com/opencv/opencv/blob/4.11.0/LICENSE and installed package LICENSE files |
| PySide6 / Qt | 6.8.3; GUI | Qt binding/module LGPL-3.0/GPL/commercial terms; https://doc.qt.io/qtforpython-6/licenses.html and https://www.qt.io/licensing/open-source-lgpl-obligations |
| PyYAML | 6.0.2; configs | MIT; https://github.com/yaml/pyyaml/blob/6.0.2/LICENSE |
| Pillow | 11.2.1; image/QC | MIT-CMU (installed 11.2.1 wheel metadata; retain bundled notices); https://github.com/python-pillow/Pillow/blob/11.2.1/LICENSE |
| ONNX Runtime | 1.22.0; user ONNX inference | MIT plus bundled notices; https://github.com/microsoft/onnxruntime/blob/v1.22.0/LICENSE |
| pytest | 8.3.5; tests | MIT; installed distribution metadata/licenses |
| Ruff | 0.11.13; lint/format | MIT; installed distribution metadata/licenses |
| pre-commit | 4.2.0; optional hooks | MIT; installed distribution metadata/licenses |
| hatchling | 1.27.0; build | MIT; installed distribution metadata/licenses |
| uv | existing 0.12.19 copied locally; pinned acquisition fallback | MIT/Apache-2.0; https://github.com/astral-sh/uv |
| Blender | detected system 4.3.2; FBX/USD subprocess | GPL; https://www.blender.org/about/license/ |
| FFmpeg / ffprobe | detected system 7.1.5; media/timestamps | This machine's GPL-enabled build is GPL; build options determine LGPL/GPL applicability; https://ffmpeg.org/legal.html |

Qt is used through separately installed, dynamically linked wheels. Redistribution must preserve notices, LGPL replacement/relinking rights and required corresponding source/access, and must avoid assuming every Qt module has identical terms. Blender is called as a separate executable; its GPL applies to distribution of Blender. FFmpeg's detected `--enable-gpl` build is not an LGPL-only binary. This repository does not bundle either system executable. Review codec patent obligations for your jurisdiction/use independently.

Wheel notices are authoritative for the exact installed binary, including BLAS, image/codec and Qt libraries. `scripts/license_audit.py` inventories installed distributions and license files and emits a review manifest. A manifest lacking an SPDX expression means manual review is needed, not permission to redistribute. The lock also contains transitive packages: retain their original license files. `docs/licenses/upstream-reviewed.json` records source references; failed outbound retrieval is explicitly marked and never presented as completed remote review. The local package review manifest is generated during final validation.

Bootstrap network: official Python package indexes to install a pinned uv wheel when no uv executable exists, then frozen lock URLs for dependencies. It does not execute downloaded shell scripts. No model download, cloud service, login or telemetry is required. Dependency wheel hashes are preserved in `uv.lock`.
