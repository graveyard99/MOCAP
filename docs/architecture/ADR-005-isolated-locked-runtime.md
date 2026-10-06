# ADR 005: Local Qt workflow and isolated locked runtime

Status: accepted for engineering preview.

A VFX artist needs one application with project management, responsive jobs, cameras, plate and scene inspection, confidence diagnostics and export. PySide6/Qt is mature, desktop-local and open-source compatible under its applicable license. UI code delegates to application services and numerical APIs; no solver is duplicated in GUI slots. Native image/video loading is asynchronous and bounded rather than loading a take into memory. QPainter-based 3D scene inspection is sufficient for the current technical fixture; dense-mesh production rendering and full data virtualization remain future work.

The installation root contains the repository, `env`, `tools`, `models`, `cache`, `downloads`, `temp` and projects. `uv.lock` pins Python dependencies. Bootstrap copies an existing uv executable locally or acquires one in a local tools environment and performs `uv sync --frozen`. Activation redirects library caches; normal launchers activate automatically. OS drivers, interpreter, display runtime and detected FFmpeg/Blender are reported separately. No global Python package installation is allowed.

Consequences: core geometry remains CPU-testable, headless CLI and offscreen UI tests share pipeline behavior, and model assets stay explicitly external. Third-party binary redistribution needs license review beyond the original Apache-2.0 source license. This preview is not a bundled cross-platform installer.
