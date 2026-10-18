# Layerwise Nuisance Accessibility Experiment Plan

> **Status:** Draft handoff for a future Vertebrae development session.
>
> **Instruction for the future session:** Read `AGENTS.md`, inspect the referenced
> implementation, and review this plan with the user before changing code. Do not
> begin implementation merely because this document exists.

## Goal summary

This study will test whether OverlapIndex (OI) can measure **nuisance
accessibility** in learned representations. Given an intended target `Y`, a labeled
nuisance factor `N`, and backbone representations `Z_l`, the study asks whether OI
can identify where nuisance structure becomes separable as network depth and
abstraction increase.

The primary claim is deliberately narrower than bias or causal reliance:

> Held-out, task-conditional nuisance OI ranks representations by how accessible a
> nuisance factor is to an OI-like downstream head.

The study will separately test whether nuisance OI predicts counterfactual reliance,
worst-group degradation, or output disparities. High nuisance OI means nuisance
structure is available in the representation; it does not, by itself, prove that the
deployed task head uses that structure or causes harm.

The initial experiment should extend
`examples/colored_fashion_mnist_shortcut.py`. That experiment already provides paired
control and shortcut models, intended and nuisance target views, intermediate layer
outputs, a balanced audit probe, repeated seeds, and counterfactual recoloring. The
expanded protocol should add logits, shortcut-strength sweeps, held-out and
null-calibrated scoring, task-conditional nuisance analysis, and formal trajectory
summaries.

## Research questions

1. Does nuisance OI track held-out nuisance decodability across models, layers,
   checkpoints, and nuisance-correlation strengths?
2. Does task-conditional nuisance OI distinguish nuisance structure from structure
   inherited through target-nuisance label correlation?
3. Where does nuisance accessibility emerge, peak, persist, or become suppressed?
4. Do layerwise trajectories explain counterfactual sensitivity or worst-group
   degradation beyond penultimate-layer or logit OI alone?
5. Are conclusions stable across prototype counts, activation dimensionality, random
   seeds, and datasets?
6. Do results transfer from controlled visual shortcuts to contextual, acquisition,
   and demographic nuisance factors?

## Operational definitions

Let:

- `X` be the input;
- `Y` be the intended task target;
- `N` be a labeled nuisance factor;
- `Z_l = f_l(X)` be the representation at layer or stage `l`; and
- `H(Z)` be the deployed task head's logits.

For each layer, measure:

```text
T_l = OI(Z_l, Y)
N_l = macro_mean_y OI(Z_l, N | Y = y)
```

`T_l` is **task accessibility**. `N_l` is **task-conditional nuisance
accessibility**. Raw `OI(Z_l, N)` should also be reported, but it is secondary because
it can reflect target-nuisance prevalence differences rather than additional nuisance
structure.

Use *accessibility* and *separability* in reports. Reserve *importance* for an
intervention or attribution analysis demonstrating an effect on task output. A
deterministic backbone cannot create nuisance information absent from the input, but
it can reorganize it so that the information becomes more or less accessible under
OI's prototype geometry.

## Hypotheses

### H1: Construct validity

Held-out task-conditional nuisance OI will positively rank representations by held-out
nuisance-probe performance across controlled conditions.

Compare OI with a fixed-capacity linear probe and one predeclared nonlinear probe. The
hypothesis concerns ranking, not equality of score scales.

### H2: Behavioral validity

Task-conditional nuisance OI, particularly at later layers and logits, will positively
associate with:

- counterfactual prediction-flip rate;
- worst-group task error;
- accuracy loss when target-nuisance correlation is removed or reversed; and
- errors that follow the nuisance-derived class.

### H3: Value of the layerwise trajectory

Peak nuisance accessibility, emergence depth, and peak-to-final suppression will
predict nuisance reliance or worst-group degradation beyond final-embedding nuisance
OI alone.

### Expected boundary

An unused nuisance code appended to an embedding should increase nuisance OI without
changing task predictions. This negative control is expected to demonstrate that
nuisance accessibility is not equivalent to causal reliance or harm.

## Phase 1: Controlled shortcut validation

Use Colored Fashion-MNIST as the first setting.

### Target, nuisance, and paired conditions

- Intended target: Fashion-MNIST garment class.
- Nuisance: displayed color.
- Control: color balanced within every garment class.
- Treatment: canonical class color with controlled probability.
- Audit probe: every garment-class by color cell exactly balanced.

Each treatment and control pair must have identical initialization, garment samples,
targets, batch order, optimizer, schedule, epoch count, and evaluation samples. Only
the class-color relationship may differ.

### Shortcut-strength sweep

Use the following predeclared canonical-color probabilities:

```text
0.10, 0.30, 0.50, 0.70, 0.85, 0.95, 1.00
```

With ten colors, `0.10` is the independence endpoint. A smaller pilot may use `0.10,
0.50, 0.85, 1.00`; run the full grid only after all controls pass.

### Named model outputs

Evaluate:

1. convolutional block 1;
2. convolutional block 2;
3. penultimate embedding; and
4. full logit vector.

The existing model already returns logits, but the extractor does not expose them in
its declared output list. Add logits as a named Vertebrae output so they are scored,
monitored, and reported like intermediate representations.

### Behavioral environments

Evaluate on:

- correlated colors matching training;
- exactly balanced colors;
- reversed canonical colors;
- grayscale/color-removed inputs; and
- exhaustive recolorings of each held-out garment.

The balanced audit probe is the primary representation-evaluation population because
it breaks target-nuisance correlation while preserving equal cell support.

## Transfer phases

### Phase 2: Contextual shortcut

Replicate on Waterbirds:

- intended target: landbird versus waterbird;
- nuisance: land versus water background;
- behavior: balanced and correlation-reversed group accuracy; and
- intervention: matched foreground-background combinations where source assets and
  licenses permit.

Use a standard pretrained vision backbone, expose major stage endpoints, embedding,
and logits, and compare ordinary ERM with at least one group-aware or
invariance-oriented method.

### Phase 3: Acquisition or domain nuisance

Prefer Camelyon17-WILDS:

- intended target: tumor presence;
- nuisance: hospital/source domain; and
- behavior: held-out-hospital and worst-domain performance.

This phase tests acquisition and processing signals rather than simple color or
background shortcuts.

### Phase 4: Demographic audit

After establishing the controlled claims, select a dataset with an intended task,
protected-group metadata, and outcome metrics. Candidates include RAF-DB, CelebA,
Multi-PIE, or BFW, subject to license and annotation review.

Requirements:

- use exact dataset label semantics, such as annotator-perceived race, self-reported
  race, skin tone, age group, or gender presentation;
- do not substitute *ethnicity* for a differently defined label;
- split repeated images by identity;
- report intersectional groups only where support is adequate; and
- retain group-specific errors and calibration alongside nuisance OI.

Describe this phase as a representation audit, not a standalone determination of
fairness or bias.

## Representation comparability

### Output metadata

Every output must record its stable name, stage, pooling rule, native dimensionality,
and any projection or compression applied before scoring.

For CNN maps, global or adaptive average pooling is the primary readout. Add a
spatially richer sensitivity analysis if pooling may erase localized nuisance signals.
For transformers, prefer the actual classification-token path, with mean token pooling
as a secondary analysis where relevant.

### Native and capacity-matched tracks

Run two scoring tracks:

1. **Native:** score the declared representation after consistent row normalization.
2. **Capacity matched:** reduce every layer to a common predeclared dimension using a
   transformation fit only on fitting rows, then normalize and score.

Record retained variance for PCA. A fixed random projection may serve as a
transformation-independent sensitivity check. The second track tests whether depth
trends are artifacts of changing activation dimensionality.

Within each replicate, reuse the same rows, folds, preprocessing boundary, OI settings,
label permutations, behavioral environments, and bootstrap units across layers.

## OI scoring protocol

### Held-out evaluation

Do not fit class-owned prototypes and make the primary claim on the same rows. Use
Vertebrae's cross-fitted OverlapIndex adapter or a separated fit/evaluation protocol
backed by `OverlapIndex.score_fixed`.

Use five-fold stratified cross-fitting when every class has sufficient support. Keep
folds paired across layers and target views.

### Scores to report

For every representation, report:

1. task OI;
2. raw nuisance OI;
3. task-conditional nuisance OI;
4. per-nuisance-class scores; and
5. pairwise scores for multiclass nuisances.

Compute conditional nuisance OI separately within each intended class, then
macro-average evaluable strata. Report insufficient-support omissions explicitly.

Evaluate separate nuisance axes separately. For intersectionality, create an explicit
joint categorical target such as `race x gender`; do not use multi-label semantics
merely because more than one attribute exists.

### Permutation null

Estimate a discrete OI null by permuting nuisance labels:

- within intended-target strata;
- at the identity, patient, or other independent-unit level;
- while preserving group sizes; and
- using the same assignments across layers within a replicate.

Report raw OI, the null mean and interval, `observed - null`, and
`(observed - null) / (1 - null)`. Use null-adjusted scores for cross-layer
differences and emergence tests while retaining raw scores for transparency.

### Prototype sensitivity

Select the primary prototype policy without test-set optimization. Run:

```text
k = 2, 4, 8, 16
```

Treat conclusions that reverse across reasonable `k` values as unstable.

## Reference diagnostics and interventions

### Nuisance probes

At every layer, fit a standardized L2-regularized linear probe and one fixed-budget
nonlinear probe. Use the same held-out folds as OI when possible. Report balanced
accuracy and, where appropriate, macro one-vs-rest AUROC. Probe results are
construct-validity references, not causal proof.

### Behavioral metrics

Report at minimum:

- overall and balanced-environment task accuracy;
- worst class-nuisance cell accuracy;
- mean accuracy of the weakest cells;
- counterfactual prediction-flip rate;
- reversed-mapping and nuisance-removed accuracy;
- errors following the nuisance-derived class; and
- group-specific FPR, FNR, calibration, or multiclass equivalents where applicable.

### Required controls

1. **Label permutation:** disconnect nuisance labels from representations.
2. **Unused code:** append an explicit nuisance code while the task head gives it zero
   weight.
3. **Nuisance removal:** remove or project out a nuisance-predictive subspace and
   re-evaluate behavior.
4. **Counterfactual input:** change only the nuisance factor where a valid matched
   transformation exists.
5. **Pretrained versus fine-tuned:** distinguish generic retained information from
   accessibility amplified by task training.

## Layer-trajectory summaries

Calculate from null-adjusted scores:

- **Emergence depth:** first layer above its paired permutation interval.
- **Peak nuisance accessibility:** maximum nuisance OI across layers.
- **Final retention:** nuisance OI at the embedding and logits.
- **Suppression:** peak nuisance OI minus final nuisance OI.
- **Task-nuisance selectivity:** task OI minus conditional nuisance OI.
- **Trajectory area:** area under nuisance OI across normalized depth.

Named stages are primary for within-model analysis. Normalized depth may support coarse
cross-architecture comparisons, but layer numbers from different architectures are
not equivalent.

## Statistical analysis

### Replication unit

The independently trained model seed is the primary unit for the controlled study. Use
at least five paired seeds for the full run, increasing this if pilot variance leaves
planned contrasts unresolved. Bootstrap repeated-observation datasets at the subject,
patient, source-image, or equivalent independent-unit level.

### Primary analysis

Estimate the association between task-conditional nuisance OI and held-out nuisance
probe performance using:

- Spearman rank correlation;
- paired bootstrap confidence intervals; and
- clustered or mixed-effects regression with model seed as the grouping factor when
  supported by the sample size.

Activation rows are not independent experimental units. Measurements from layers of
the same trained model must remain grouped.

### Secondary analyses

1. Predict counterfactual flip rate and worst-group degradation from final nuisance OI.
2. Add peak OI, emergence depth, and suppression to test incremental out-of-sample
   value from the trajectory.
3. Estimate paired treatment-control effects at every named layer.
4. Compare fine-tuned and pretrained nuisance accessibility.
5. Test agreement between native and capacity-matched tracks.

Use simultaneous bootstrap intervals or false-discovery-rate control for layerwise
contrasts. Mark post hoc analyses as exploratory.

## Success and falsification criteria

### Minimum supported claim

Support nuisance OI as an accessibility diagnostic if:

1. conditional nuisance OI exceeds its permutation null in known nuisance-encoding
   treatments;
2. it consistently ranks held-out nuisance-probe performance;
3. the unused-code control raises OI without changing behavior; and
4. conclusions remain stable across seeds, reasonable `k` values, and the
   capacity-matched track.

This supports accessibility, not causal reliance.

### Stronger supported claim

Support OI as a shortcut-risk indicator if, in addition:

1. nuisance OI predicts counterfactual sensitivity or worst-group degradation;
2. logit nuisance OI is more strongly associated with behavior than early-layer OI;
3. trajectory summaries add value beyond the final embedding; and
4. the relationship replicates on a real contextual or acquisition dataset.

### Limiting outcomes

Narrow or reject the stronger claim if:

- nuisance OI tracks probes but not behavior;
- final OI is sufficient and trajectories add no reproducible information;
- rankings reverse under modest `k` or dimensionality changes;
- OI remains at null while strong nuisance probes succeed; or
- demographic results fail to replicate across datasets or identity-disjoint splits.

These outcomes are informative because they define the diagnostic's boundary.

## Reproducibility requirements

Record for every run:

- repository commit and package versions;
- dataset identity, version, license, and split construction;
- model architecture, initialization, and checkpoint;
- optimizer, schedule, epoch, and global step;
- output name, stage, pooling, and native dimension;
- projection or compression recipe;
- OI configuration, `k`, normalization, and seeds;
- folds and nuisance-label permutations;
- target-view definitions and omitted strata;
- intervention parameters; and
- wall-clock and peak-memory measurements where practical.

Use `RepresentationMonitor` for depth-by-training-time records. Mark generated
artifacts as pilot, exploratory, interrupted, or final.

## Planned artifacts

The controlled study should produce:

1. a machine-readable run manifest;
2. seed-level long-form results;
3. aggregate estimates with confidence intervals;
4. a depth-by-shortcut-strength nuisance OI heatmap;
5. paired task and nuisance trajectories;
6. OI-versus-probe and OI-versus-counterfactual plots;
7. an output-only versus trajectory-model comparison; and
8. a concise audit summary stating both findings and non-claims.

Keep raw datasets, checkpoints, and large activation stores outside Git. Check in
protocol definitions, small summaries, tests, and final documentation figures where
appropriate.

## Repository ownership

This work belongs in Vertebrae because it requires dataset protocols, multi-output
extraction, target views, monitoring, probe comparisons, counterfactual evaluation,
and reporting.

Upstream only generally useful metric capabilities to OverlapIndex, such as discrete
permutation-null calibration, generic held-out scoring improvements,
backend-independent corrections, or finite-sample and dimensionality tests.
OverlapIndex should not acquire backbone integrations, layer concepts,
dataset-specific audits, or causal-behavior claims.

## Initial implementation checklist

- [ ] Add logits to the Colored Fashion-MNIST named outputs.
- [ ] Add logit-output regression coverage.
- [ ] Add pilot and full shortcut-strength grids.
- [ ] Use cross-fitted OI for the primary path.
- [ ] Add conditional nuisance aggregation with explicit support checks.
- [ ] Add paired, stratified nuisance-label permutations.
- [ ] Add linear and nonlinear nuisance-probe references.
- [ ] Add native and common-dimensional tracks.
- [ ] Calculate trajectory summaries and paired seed effects.
- [ ] Extend plots with logits and null intervals.
- [ ] Run the controlled pilot and freeze the protocol.
- [ ] Replicate on Waterbirds.
- [ ] Replicate on an acquisition/domain dataset.
- [ ] Complete dataset governance review before demographic work.

## Decision gates

1. **Pipeline:** all controls behave as expected on the pilot grid.
2. **Construct validity:** nuisance OI ranks held-out nuisance decodability.
3. **Trajectory:** layerwise summaries reproduce and add information beyond the
   embedding or logits.
4. **Behavior:** nuisance OI or its trajectory predicts counterfactual or worst-group
   failure.
5. **Transfer:** findings replicate outside the synthetic color shortcut.
6. **Demographic:** only after earlier gates pass should a demographic application
   become the central example.

Passing an earlier gate does not establish later causal or fairness claims.
