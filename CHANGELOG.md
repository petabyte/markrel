# Changelog 🐟

All notable changes to markrel are documented here.
This project follows [Semantic Versioning](https://semver.org/).

## v0.2.0 (2026-09-25)

### Added
- `markrel.integrations.JevCalibrator` — domain recalibration for Jev (or any
  external) relevance probabilities. Learns `P(actually relevant | scorer said p)`
  from your own ground-truth labels using markrel's histogram-binning Markov
  chain, with quantile or uniform binning and Laplace smoothing.
- `docs/jev-calibrator.md` — usage guide for the calibrator.
- Documentation site published to GitHub Pages, plus social card assets.

### Changed
- README covers `JevCalibrator` alongside the core model.

## v0.1.0 (2026-03-24)

### Added
- Initial release: Markov chain model for document relevance prediction.
- `MarkovRelevanceModel`, `MetricChain`, and supporting transition machinery.
- CI and PyPI trusted-publishing workflows.
