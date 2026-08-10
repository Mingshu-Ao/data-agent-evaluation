# Public Repository Policy

## Included

- Baseline source code and tests
- Example configuration with environment-variable API key references
- Task suite metadata
- Synthetic success and failure output examples
- Integration protocol and documentation

## Excluded

- Raw Benchmark inputs and gold answers
- `configs/*.local.yaml`
- API keys, account credentials, VPN and SSH settings
- `.venv`, caches and model downloads
- Runtime `artifacts`, full traces and generated predictions
- OCR, ASR and video preprocessing caches

Team members should obtain datasets through the authorized source and place them in the
local paths documented in `README.md`.
