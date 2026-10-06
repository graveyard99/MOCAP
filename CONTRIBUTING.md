# Contributing

Bootstrap with `scripts/bootstrap.sh` and source `scripts/activate.sh`. Development dependencies are locked in `uv.lock`; changes to `pyproject.toml` must update and review the lock. Work on a branch and run `ruff check .`, `ruff format --check .` and `pytest`. UI tests use `QT_QPA_PLATFORM=offscreen`; export tests additionally need Blender. Install optional local Git hooks with `pre-commit install` after activation.

Preserve the evidence hierarchy: locked manual/calibrated geometry, strong multiview evidence, temporal observations, body constraints, biomechanics, learned priors, monocular estimates. Camera and time parameters must carry provenance and explicit ownership. Never overwrite raw observations or authoritative calibration. New solver terms need inspectable residuals, confidence handling and synthetic adversarial tests.

Changes should include meaningful tests, user-facing documentation and an honest update to `docs/BUILD_STATUS.md`. Test external integrations with deterministic fixtures and clearly distinguish a fixture from real weights or licensed assets. Do not add checkpoints, footage, generated results, environments, secrets or proprietary model files to Git. Review source, binary and checkpoint licenses separately before a dependency is introduced.

Report geometry bugs with coordinate convention, camera count, timestamps, config, seed, QC residuals and a small redistributable fixture. For private material, reduce the problem to synthetic data. PRs should describe the concrete before/after behavior, relevant numerical bounds and actual test commands/results.
