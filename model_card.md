# Model Card — IRIS Species Classifier

**Version:** 1.0 · **Last updated:** Week 9 submission · **Owner:** [Yashvardhan | 23f2004644]

## Intended Use
Classifies iris flowers into one of three species — setosa, versicolor,
or virginica — based on four physical measurements. Intended as a
teaching/demonstration model for an MLOps course pipeline (data
versioning, experiment tracking, fairness auditing, explainability, and
drift monitoring). **Not intended for any production botanical,
agricultural, or commercial classification use** — the dataset is a
small, classic benchmark, not a representative real-world sample.

### Out-of-Scope Uses
- Any deployment decision affecting real people, resource allocation, or
  access to services — this model was never designed or validated for
  human-subject or demographic decision-making.
- Any use of the `location` attribute as a predictive input; it exists
  solely as an audit variable and must never be added back into the
  feature set.
- Any production use without re-validating on real, larger-scale,
  domain-representative data first.


## Model Details
- **Algorithm:** RandomForestClassifier (scikit-learn), 100 estimators
- **Training features:** sepal length, sepal width, petal length, petal width (cm)
- **Excluded from training:** `location` — a randomly assigned sensitive
  attribute used only for fairness auditing, never as a predictive feature
- **Target:** 3-class species label (setosa, versicolor, virginica)

## Training Data
- Source: the standard scikit-learn IRIS dataset (Fisher, 1936), 150 samples,
  50 per class, perfectly balanced
- A `location` column (0 or 1) was added and randomly assigned independent
  of species and features, simulating a demographic-style group label for
  fairness-audit practice
- Train/test split: 80/20, stratified by species, fixed random seed (42)

## Performance

**Overall (held-out test set, n=30):**
| Metric | Value |
|---|---|
| Accuracy | 0.900 |
| Precision (macro) | 0.902 |
| Recall (macro) | 0.900 |

**Disaggregated by `location` group:**
| Group | Accuracy | Precision | Recall |
|---|---|---|---|
| location = 0 | 0.846 | 0.878 | 0.878 |
| location = 1 | 0.941 | 0.933 | 0.933 |

Max accuracy gap between groups: ~9.5 percentage points.

## Limitations
- **Small test set (n=30):** the observed ~9.5-point accuracy gap between
  location groups is very likely sampling noise rather than genuine bias,
  since `location` was randomly assigned and has no causal relationship
  to species or features. A larger test set would be needed to confirm
  this gap disappears, as expected under a truly random attribute.
- **Dataset size and diversity:** 150 samples is small by modern ML
  standards; the classes are also perfectly linearly/near-linearly
  separable, which is not representative of most real-world classification
  problems.
- **No adversarial robustness testing** beyond the poisoning simulation
  done in a separate assignment — this model has not been evaluated
  against adversarial inputs at inference time.
- **Drift sensitivity:** feature-distribution drift (see drift analysis)
  on petal length/width was statistically significant after a simulated
  shift; a model trained on the original distribution would likely see
  degraded accuracy against such shifted production data, since it has
  never seen inputs in that range during training.

## Fairness Considerations
- `location` was deliberately excluded from training features to avoid
  using demographic-style information as a predictor.
- Fairlearn's `MetricFrame` was used to audit — not adjust or correct —
  performance disparities across the `location` groups. No mitigation
  (e.g. reweighting, threshold adjustment) was applied in this exercise;
  the audit itself is the deliverable.
- In a real production setting with a genuine sensitive attribute (not a
  randomly assigned one), any observed performance gap of this size
  would warrant further investigation before deployment, since it could
  reflect real disparate impact rather than sampling noise.

## Explainability
SHAP analysis (TreeExplainer) shows petal length and petal width are by
far the most influential features for all three classes, consistent with
the known biological separability of IRIS species by petal dimensions.
Sepal width contributes comparatively little to any class prediction.

## Monitoring & Maintenance Plan
- **Drift monitoring:** Statistical distribution tests (Kolmogorov–Smirnov)
  should be run periodically comparing live inference inputs against the
  original training distribution for each feature. In this assignment's
  simulation, a KS test correctly flagged petal length and petal width as
  drifted (p < 0.001) after an artificial shift, while sepal features
  correctly showed no drift (p = 1.0) — this is the same check that
  should run against real production traffic.
- **Concept drift:** Since KS tests on inputs alone cannot detect concept
  drift, ground-truth labels (or a proxy/delayed feedback signal) should
  be collected on a sample of predictions and periodically compared
  against model accuracy to catch cases where the input-output
  relationship itself has changed.
- **Retraining trigger:** A recommended policy is to flag for review any
  feature whose KS test p-value falls below 0.05 against a rolling
  production window, and to retrain once cumulative drifted features or
  a measured accuracy drop exceeds a predefined threshold.
- **Fairness re-audit cadence:** Any retrained model should be re-run
  through the same Fairlearn `MetricFrame` audit before redeployment, so
  a performance gap introduced by retraining is caught before release.