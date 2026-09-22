# JevCalibrator

**Domain recalibration for Jev, built on markrel's Markov-chain binning.**

`JevCalibrator` is an integration in [`markrel`](https://github.com/petabyte/markrel) (`markrel/integrations/jev_calibrator.py`, merged in [PR #1](https://github.com/petabyte/markrel/pull/1)) that recalibrates the probability Jev outputs so it reflects the actual base rate in *your* domain, instead of Jev's generic judgment of the question it was asked.

---

## The problem it solves

Jev takes a piece of text and a typed question and returns a calibrated probability. That calibration is with respect to Jev's own judgment task — it is not automatically calibrated to whatever downstream decision you're using it for. Two things break the naive "just threshold Jev's score" approach:

1. **The score means something different depending on the question.** Change the criteria you ask Jev to judge, and the same document can produce a different probability, because the question is part of the model.
2. **Jev's stated confidence and your domain's true hit rate can diverge.** A score of `0.9` doesn't mean "90% of the time this is correct" for your specific corpus, labeling process, or definition of relevant — it means "Jev's internal judgment process landed here," which may be systematically over- or under-confident relative to your ground truth.

This is the same failure mode that motivates calibration techniques like Platt scaling or isotonic regression for any classifier: a model's raw score can rank well while still being off in absolute terms. `JevCalibrator` applies that same fix, using markrel's existing bin-based Markov chain as the mechanism.

## How it works

`JevCalibrator` is a thin wrapper around `markrel.MetricChain`. Instead of binning a cosine or Euclidean similarity score (markrel's usual input), it bins Jev's raw probability directly:

```
Jev score → quantile bin → P(actually relevant | this bin), learned from your labels
```

You fit it once on a set of `(jev_score, true_label)` pairs, and it learns a bin table you can then use to convert any new Jev score into a recalibrated one. It never touches text or embeddings — it operates purely on the scalar scores, so it's a standalone calibration layer that sits *after* Jev in your pipeline, not a replacement for it.

Because it's built on `MetricChain`, the same machinery markrel uses to explain its own predictions — bin edges, per-bin counts, per-bin probability — is available for auditing exactly how the recalibration behaves at each part of the score range.

## Sample usage

### Basic recalibration

```python
import numpy as np
from markrel.integrations import JevCalibrator

# jev_probs: Jev's raw P(relevant) output for each training pair
# labels: independent ground-truth relevance labels for the same pairs
#         (NOT produced by Jev itself — see the caveat below)
jev_probs = np.array([0.91, 0.83, 0.4, 0.12, 0.77, 0.55, 0.68, ...])
labels    = np.array([1,    0,    0,   0,    1,    0,    1,    ...])

cal = JevCalibrator(n_bins=20, bin_strategy="quantile")
cal.fit(jev_probs, labels)

# Recalibrated probability for new Jev scores
cal.predict_proba([0.83])
# array([0.61])   <- Jev said 0.83; in this domain that band is actually ~61% hit rate

# Binary decision using the recalibrated probability, not the raw one
cal.predict([0.2, 0.5, 0.8, 0.95], threshold=0.5)
# array([0, 0, 1, 1])
```

### Inspecting the learned calibration curve

```python
cal.summary()
# {
#   "metric": "jev_prob",
#   "n_bins": 20,
#   "total_relevant": 984,
#   "total_not_relevant": 1016,
#   "states": [
#     {"bin": 0, "range": [0.0, 0.08], "n": 134, "p_relevant": 0.066},
#     {"bin": 1, "range": [0.08, 0.18], "n": 133, "p_relevant": 0.104},
#     ...
#   ]
# }

cal.bin_edges()
# [(0.0, 0.08), (0.08, 0.18), ...]
```

This is the whole point of using markrel's approach rather than a black-box calibration method — you can see exactly which score ranges Jev is over- or under-confident in for your domain, not just get a single corrected number back.

### Cascade: cheap filter first, Jev only for the ambiguous middle

```python
from markrel import MarkovRelevanceModel

# A fast, local, embedding-based first pass
embed_model = MarkovRelevanceModel(metrics=["cosine"], n_bins=20)
embed_model.fit(train_queries, train_docs, train_labels)

# Recalibrated Jev, used only on the borderline cases the cheap model can't resolve
cal = JevCalibrator(n_bins=20, bin_strategy="quantile")
cal.fit(jev_probs_on_borderline_cases, labels_on_borderline_cases)

combined = cal.combine_with(
    jev_probs, embed_model, queries, documents, rule="bayesian"
)
```

`combine_with` merges the recalibrated Jev probability with a markrel embedding-based model's probability using the same Bayesian odds-product rule markrel uses internally to combine its own metrics — so a fast, cheap similarity signal and a recalibrated Jev judgment can vote together on the final score.

## How this solves domain adaptation for Jev

Jev's steerability comes from rewriting its instructions — you adapt it to a new domain by changing the question you ask, not by retraining a model. That's powerful, but it leaves a gap: rewriting the instruction changes *what* Jev is judging, not *how well its stated confidence matches your domain's actual outcomes*. Two domains can use the identical Jev question and still have different true base rates, different cost structures for false positives vs. false negatives, and different distributions of "hard" borderline cases — none of which Jev's prompt-level steering corrects for.

`JevCalibrator` closes that gap without touching Jev itself:

- **No retraining or reprompting of Jev.** You keep whatever instruction/criteria you've already tuned for the domain. The calibrator sits entirely downstream, correcting the mapping from Jev's score to your domain's true probability.
- **Adapts with a small amount of labeled data.** Fitting the bin table only needs `(jev_score, true_label)` pairs — no embeddings, no text, no domain-specific feature engineering. This is far cheaper than adapting via a full classifier.
- **Makes thresholds portable across domains.** A fixed threshold like "flag if Jev says ≥ 0.7" silently means something different in every domain Jev is deployed in. Calibrating first means the same downstream threshold (e.g. 0.5 on the *recalibrated* probability) behaves consistently.
- **Transparent, inspectable adaptation.** `summary()` and `bin_edges()` show exactly how the domain's true rate diverges from Jev's raw score at every point in the range, so the adaptation itself is auditable rather than a black box.
- **Composable with markrel's own domain-adapted models.** Via `combine_with`, a recalibrated Jev signal can be blended with a cheap embedding-based model trained on the same domain, letting the cheap model absorb the bulk of the volume and Jev's (now properly calibrated) judgment weigh in on the harder cases.

## Important caveat: independent labels only

`JevCalibrator` must be fit on labels that are **independent of Jev** — human review, click-through data, downstream outcomes, anything Jev didn't produce. If the training labels were themselves generated by Jev (e.g. as a weak-supervision bootstrap), calibrating Jev against Jev-derived labels just reproduces Jev's own scores and adds no real correction.

## Reference

- Source: [`markrel/integrations/jev_calibrator.py`](https://github.com/petabyte/markrel/blob/master/markrel/integrations/jev_calibrator.py)
- Tests: [`tests/test_jev_calibrator.py`](https://github.com/petabyte/markrel/blob/master/tests/test_jev_calibrator.py) (10 tests covering fit/predict, recalibration behavior, input validation, and `combine_with`)
- Merged via [PR #1: "Add JevCalibrator: domain recalibration for external relevance scorers"](https://github.com/petabyte/markrel/pull/1)

**API summary**

| Method | Description |
|---|---|
| `fit(jev_probs, labels)` | Learn the recalibration curve from raw Jev scores and independent ground-truth labels |
| `predict_proba(jev_probs)` | Return recalibrated `P(relevant)` for new Jev scores |
| `predict(jev_probs, threshold=0.5)` | Binary prediction, thresholded on the *recalibrated* probability |
| `combine_with(jev_probs, other_model, queries, documents, rule="bayesian")` | Combine recalibrated Jev scores with a markrel embedding-based model |
| `summary()` | Bin-by-bin breakdown: Jev's raw score range vs. the domain's true relevance rate |
| `bin_edges()` | The `(lo, hi)` edges of each calibration bin |
