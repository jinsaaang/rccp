# Method Implementation

This folder contains the active method and forecaster implementation.

```text
models/             forecasting and uncertainty models used by the Hydra runner
upstream/hopcpt     HopCPT upstream snapshot for reference/golden checks
upstream/rescp      ResCP upstream snapshot for reference/golden checks
upstream/ct_ssf     CT-SSF upstream snapshot for reference/golden checks
```

New benchmark code should import from `models.*` through the root compatibility
package, which points at `methods/models`.
