# ML prioritization

The canonical documentation for each trained-model cohort lives **next to
the artifacts**, not in `docs/`. This file is just a pointer index.

## Current cohort

- **20260513** — 52 per-gene Random Forest regression models (CV R² > 0.1,
  H236 multi-FP features, single RF). See
  [`output/ML/trained_models/20260513/README.md`](../output/ML/trained_models/20260513/README.md)
  for the full description (data sources, filters, features, training pipeline,
  bundle schema, reproducibility recipe).

## Why the docs live with the artifacts

Each `output/ML/trained_models/<date>/` folder is a self-contained model
registry — the `*.joblib` bundles and a `README.md` describing how they were
produced sit side-by-side. This keeps documentation discoverable from the
folder a collaborator opens to grab a model, and avoids stale docs in
`docs/` getting out of sync with newer cohorts.

Note: `output/` is gitignored (heavy / private artifacts), so these READMEs
are not version-controlled. If you need the doc in git for audit or sharing,
copy the relevant cohort README into `docs/ML_prioritization_<date>.md`.

## Related

- [`SAR_prioritization.md`](SAR_prioritization.md) — methodology / why CV R² is the right per-gene predictability proxy.
- [`documentation.md`](documentation.md) — the four-notebook flow that produces these bundles.
