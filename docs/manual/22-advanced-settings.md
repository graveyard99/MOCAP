# Advanced settings and solver governance

The toolbar exposes documented presets. They currently control robust triangulation hypothesis/iteration budgets:

| Preset | Maximum hypotheses | Maximum iterations |
| --- | ---: | ---: |
| Preview | 16 | 40 |
| Standard | 32 | 80 |
| High Quality | 64 | 100 |
| Maximum / Final | 128 | 200 |

They do not silently choose different neural checkpoints, redefine scale or unlock surveyed cameras. More iterations cannot repair wrong lens metadata or cross-person observations.

Camera / lens and Transform expose normal measured parameters. Expert values allows complete canonical records. The inspector's Expert override controls call the same audited service as the normal UI. Parameter keys must identify actual supported project fields; arbitrary extra metadata does not automatically create a solver objective.

Root/body output, solver bounds and optional inference codecs have configuration-level controls for technicians. Strong-evidence displacement defaults to a bounded tolerance, with diagnostic reporting. The body fitter's tiny rotational initializer resolves unobservable twist; it is not allowed a comparable weight to strong geometry.

The current UI provides progressive disclosure for actor/inference and basic/expert camera values, but no comprehensive Basic/Advanced/Expert numerical solver dashboard. Use the technical reference for API/config details, and confirm objective contributions in QC before adopting new numerical settings.

Cached stages are reused only when inputs, settings, schema and code fingerprints remain compatible.
