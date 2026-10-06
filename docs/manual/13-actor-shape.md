# Actor shape

Open Actor → Actor and model. Enter actor name, selected body-model family, optional measured height and licensed numerical model file. Apply actor setup before fitting. The fixture model is a technical synthetic mannequin and must not be presented as an actual SMPL asset.

![Actor and persistent proportions](assets/actor-workspace.png)

Fit persistent shape… uses observations accumulated across the take to determine one shape solution. Body pose can change each frame; fitted shape/rest proportions remain constant. The table displays fitted femur, tibia, arm, torso and width measurements when the loaded rig exposes corresponding joints.

The current shape fit primarily uses stable measured bone lengths. Silhouette envelope, body volume, direct manual-measurement objectives and known-height constraints are incomplete. The provided height field is retained metadata and comparison information; it does not silently change geometric scale.

A missing SMPL-family file fails explicitly. The program never changes smpl to fixture automatically. Use the assets specified in the project, with licensing rights suitable for the intended work.

If fitted proportions look wrong, inspect original joint mapping, identity, calibration, timing and occlusion before changing shape regularization. Compare several poses and views; a single foreshortened frame should not determine persistent body shape.
