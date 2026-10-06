# First launch

Launch OpenMocap.sh. The project manager offers New Project, Open Project, recent projects, Synthetic Example / Tutorial, Project Health / Doctor and Documentation.

On first launch, the application runs local diagnostics and presents Project Health / first-run setup. Review the report before importing footage. Missing optional model files are expected on a clean installation; missing numerical dependencies or write permissions need repair.

Use Locate licensed model assets… to choose your own numerical NPZ body-model file. Model setup manual opens relevant help. Synthetic example creates the tutorial without licensed assets. Close dismisses the diagnostic window. The application remembers that this setup has been shown; Help → Environment Diagnostics and Project → Project Health / Doctor rerun it later.

![Project manager](assets/project-manager.png)

Recent projects and viewer preferences are stored in installation-local cache/ui/preferences.json. There is no cloud account requirement. Diagnostic results report the actual detected runtime rather than assuming that an installed GPU is visible to Python.

For a first result, follow the [quick start](quick-start.md). If an optional backend is unavailable, the application retains working geometry and reports the external requirement instead of fabricating an export or inference result.
