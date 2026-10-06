# Body assets and inference-model licenses

No pretrained checkpoint, SMPL-family file, capture footage or third-party body coefficients are redistributed. Source license and weight license are distinct. A backend interface is not evidence that a model checkpoint is licensed for commercial VFX.

| Asset | Current integration | Acquisition / license requirement |
|---|---|---|
| SMPL | numerical coefficient/LBS adapter | Obtain from https://smpl.is.tue.mpg.de/ and accept the applicable license; commercial use may require a separate agreement |
| SMPL-H | full joint rotations supported | Obtain authorized SMPL-H/MANO assets and inspect their applicable agreements |
| SMPL-X | full joint rotations supported, no expression/PCA convenience | Obtain from https://smpl-x.is.tue.mpg.de/ and inspect the applicable model license |
| Compatible pose ONNX | local heatmap/SimCC/coordinate runtime contract | User supplies checkpoint, joint order, preprocessing and license; actual production checkpoint quality has not been validated |
| Compatible detector ONNX | local detector runtime contract | User supplies compatible model and verifies source/weights/redistribution terms |
| Compatible segmentation ONNX | local interface/runtime | User supplies approved checkpoint; silhouette shape objective is not implemented |
| Technical fixture | generated synthetic weighted cylinder mesh | Original project fixture; Apache-2.0, explicitly selected as `fixture`, never a SMPL substitute |

Expected isolated paths: `<INSTALL_ROOT>/models/body/SMPL_NEUTRAL.npz`, analogous explicitly selected SMPL-H/SMPL-X files; `<INSTALL_ROOT>/models/pose/*.onnx`; `<INSTALL_ROOT>/models/segmentation/*.onnx`. The actual model path is configurable through the wizard/Actor panel and config. Startup validates the model contract; missing or malformed required assets fail loudly. See [body model contract](smpl/body-model-contract.md) for arrays, joint counts and conversion.

Numerical NPZ loading avoids arbitrary pickle execution. Official legacy pickle conversion is explicitly opt-in (`trust_pickle=True`) and must only process trusted, legally acquired files in a controlled environment. Do not load an unknown pickle. Do not upload/commit licensed converted assets. Actual licensed-model capture validation remains BLOCKED_EXTERNAL until the owner supplies the assets.

The personal-experiment preset selects author-hosted RTMW-l distilled whole-body
and YOLOX-m HumanArt ONNX exports. Official source terms and deployment contracts
were reviewed; actual download and real-weight inference remain unverified because
of the sandbox network permission blocker. See [public asset provenance and
setup](perception/PUBLIC_MODELS.md). Code terms, training-data terms and weight
redistribution rights remain distinct; checkpoints are never bundled into Git.
No incompatible or lower-detail body is selected silently.

## Personal experimentation and official body assets

Reviewed on 2026-10-06: the operative grants on the official
[SMPL model licence](https://smpl.is.tue.mpg.de/modellicense.html) and
[SMPL-X model licence](https://smpl-x.is.tue.mpg.de/modellicense.html) include
non-commercial research, education and artistic projects. Personal artistic
experimentation fits a listed purpose, subject to the remaining agreement terms.
The [SMPL](https://smpl.is.tue.mpg.de/) and
[SMPL-X](https://smpl-x.is.tue.mpg.de/) portals still require the owner to register
and agree to the applicable download licence. Public source code does not include
the numerical body coefficients. The project does not need an account password;
the owner can obtain the model archive and place its selected asset beneath
`<INSTALL_ROOT>/models/body/`.

Start with a neutral SMPL asset for body-only validation, or an official
`SMPLX_NEUTRAL.npz` for the extended joint model. SMPL-X's safe numerical NPZ avoids
legacy Chumpy pickle conversion. Choosing SMPL-X does not enable unfinished face,
expression or detailed hand-fitting objectives. Preserve the source archive,
its licence and its provenance privately; do not commit full model coefficients.

Full model coefficients and an exported fitted character have distinct terms.
The official [SMPL Body](https://smpl.is.tue.mpg.de/bodylicense.html) and
[SMPL-X Body](https://smpl-x.is.tue.mpg.de/bodylicense.html) pages describe a
CC-BY-4.0 body subset that excludes the shape blendshapes and shape-generation
tools. An export's actual contents determine whether it belongs to that subset;
the full model archive must not be described as CC-BY-4.0. Provide the required
attribution when sharing qualifying body outputs.
