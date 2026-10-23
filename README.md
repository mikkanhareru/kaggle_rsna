# key_idea — the code behind our two final RSNA knee-MRI submissions

Curated on 2026-10-20 from the working repo (`RSNA_kaggle/`). Only code that the two final submissions
actually depend on is kept; every experiment whose gate failed (LABCAL label calibration, anatomical mirror,
published-HP constants, RN50D-on-d96 "RND96", VLM/LLM synovitis labels, ...) is left out.
`MANIFEST.sha256` maps every file here to its source and proves the copies are byte-identical.

| Track | Kaggle notebook | Public LB | Runtime | Ranking |
|---|---|---|---|---|
| **Accuracy** | `rsna-knee-dread-d96-ssl` (Version 3) | **0.936** | 3h| - |
| **Efficiency** | `rsna-knee-eff-d96-k43-2gpu` (Version 2) | **0.933** | **~12 min** |**144/6275**|


## Lineage

```
public "Dread" 0.924 notebook (read, re-implemented; reference source kept for its preprocessing)
  -> R      our reconstruction, our weights, public CC0 labels                         LB 0.928 / 0.929
  -> RGR    + DropPath 0.1 + pairwise rank loss (lambda 0.5, tau 1.0)                   LB 0.929-0.932
  -> d96    DENSE96 corpus: 96 slices at native 384 px, K_EVAL 86                       LB 0.935
  -> d96-ssl  0.4 rank(d96 s42) + 0.4 rank(d96 s43) + 0.2 rank(SSL member)              LB 0.936  <- accuracy pick

SSL member "s4mae": DINOv2 ViT-S/14 -> continued MAE pretraining on competition images (folds 0-3,
step 150k) -> ep50nb supervised fine-tune, 5 folds, 336 px slot cache. Decorrelated from CoAtNet (rho ~0.82).

EFF (efficiency): the d96 s42 checkpoint alone, K_EVAL 43, DICOM preparation in a fork()ed process pool
(6 workers + 8-thread page-cache prefetch), one model replica per T4 (the Kaggle box has two)
  ~2 h / 0.935  ->  ~20 min / 0.933 (v1)  ->  ~12 min / 0.933 (v3)   <- efficiency pick
```

## Layout

| Directory | Contents |
|---|---|
| `rsna/` | Shared Python package (namespace package, no `__init__.py`): data preparation, label building, model, training loop, MAE pretraining. Run modules as `python -m rsna.<...>` from this directory. |
| `preprocessing/` | How to build every corpus/cache/label file the models train on, in order; the reference inference source the corpus builders extract from. |
| `public_LB_model/` | The 0.936 accuracy model: CoAtNet-2 d96 trainer + launcher, SSL member launchers, the Kaggle kernel, bundle metadata. |
| `eff_LB_model/` | The 0.933 / ~12 min efficiency kernel and the builder that derives it from the scored single-arm d96 kernel. |
| `utils/` | Verification (byte-parity, failure injection, fork-safety) and measurement (K_EVAL curve, Kaggle timing probes, submission poller). |
| `docs/` | The preregistrations behind each component, with their frozen rules and results. |

## Caveats

- **No weights or data are stored here.** Weights live in the private Kaggle datasets listed in
  `public_LB_model/bundles/`; competition data stays on the server since rules forbid redistribution.
- The SSL member's MAE encoder was pretrained on competition images; see `public_LB_model/bundles/slot_s4mae/PROVENANCE.md`.
