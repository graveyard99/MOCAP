# Body solving

After reconstructing metric joints and fitting persistent shape, choose Solve body motion… in Actor or Run Pipeline… in the toolbar.

The fitter solves root/global translation and local joint rotations with fixed shape and hierarchy. It supports confidence-weighted 3D joints, direct calibrated 2D reprojection when configured, and a small rotational initialization term for unobservable twist. Strong geometric evidence has a displacement guard; camera calibration is not optimized as a hidden body variable.

The output includes a conventional skeleton, a constant-topology rest mesh, skin weights, local rotations and global motion. The 3D viewer skins the actual fitted mesh rather than showing an unrelated frame-by-frame mesh sequence.

A measured joint position does not fully determine axial twist. Neutral/template initialization of unobserved asset joints is recorded and carries no fabricated measurement confidence. Hand/face-specific fitting, anatomical joint-limit objectives, learned temporal priors and comprehensive twist recovery remain incomplete.

Check geometric and fitted-body reprojection separately in the HTML QC report. Excellent triangulation can coexist with a poor rig fit; the report exposes both.

Persistent shape is preserved during contact or kinematic re-fitting through the reference animation. Solver failure retains diagnostics and completed checkpoints. Retry reruns through the shared application service; it does not change the evidence hierarchy.
