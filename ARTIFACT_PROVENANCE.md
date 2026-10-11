## Executive summary (read this first)

This participant is an MIT-licensed, MindTS-inspired API forecasting implementation.
It contains no pretrained local models, fitted model files, competition task data or outcomes.
The runtime House model is supplied by the organizer, not distributed in this image.

## Implementation and data

- Numerical forecasting uses stationary block bootstrap paths and a Student-t fallback.
  Parameters are computed from the current unit's observations at or before its as-of date.
  No learned parameter files are bundled. These are algorithms, not pretrained models.
- Text reasoning uses one House chat request. Evidence requires a verbatim corpus quote,
  a matching asset patch and a valid pre-cutoff timestamp. Inferred evidence is shrunk.
- MindTS motivates the endogenous/exogenous alignment and condensation design only.
  This is not a reproduction of its anomaly-detection architecture, hidden-state access
  or pretrained weights. No MindTS source code or weights were copied.
- `horizons.py` and `targets.py` adapt MIT-licensed helpers from the official Track 2 source
  at commit `a2c93e7b66b8c6a6c429fda81d6bc7873b7c5dec`.
  They resolve observation-month mappings and convert decimal simple returns with log1p.
- The inherited public Track 2 package remains MIT-licensed. Its scoring rules are unchanged.
  Shared scoring and proof routines are imported from the official toolkit, never rewritten.

## Dependencies and model disclosure

The image installs Python 3.13, NumPy 2.1.3, pandas 2.2.3, PyArrow 18.1.0,
jsonschema 4.23.0, and official `qfbench2-common` from release v2.6.0.
Dependency license notices remain in their installed distributions. The toolkit's license
and competition conditions apply separately; this file does not relicense dependencies.
`/opt/LICENSE` contains the participant package's MIT license.

The descriptor reports the organizer's House identity, rather than the runtime routing alias:

```json
{
  "name": "nvidia/nemotron-3-super-120b-a12b",
  "version": "rl-030326-fp8",
  "revision": "rl-030326-fp8",
  "training_cutoff": "unpublished",
  "access": "api"
}
```

This is the organizer's published disclosure, not an independently inspected model snapshot.
No training-cutoff date is invented. The container obeys the injected `MODEL_NAME` routing alias.

## Release binding

The manual release workflow records the Git commit in OCI image labels and `image-metadata.json`.
It uses that built image's actual registry digest in the Development descriptor draft.
The final descriptor digest and team proof are produced by the official v2.6.0 packer.
No Team Key is present in the source, image, workflow or release artifacts.
An actual immutable image digest is known only after the image has been built and pushed.
