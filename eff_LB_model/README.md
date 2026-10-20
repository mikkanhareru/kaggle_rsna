# eff_LB_model — EFF v3, public LB 0.933 in ~12 min

Efficiency score = AUC / (Benchmark - maxAUC) + RuntimeSeconds / 32400 (lower is better);
1 minute of runtime costs about as much as 0.0009 AUC. Full record: `../docs/preregistered/2026-10-19_EFF_efficiency_track.md`.

## kernel/
`rsna_dread_eff_d96_k43_2gpu.py` = the scored single-arm d96 kernel (LB 0.935, ~2 h) with:
1. **K_EVAL 86 -> 43** (K-curve: AUC flat until coverage drops below ~1/3; LB cost 0.002).
2. **Process-pool DICOM preparation**: 6 fork()ed workers, each first reads the study's selected series into the
   page cache with 8 threads, then runs the UNCHANGED `build_study` (bytes identical by construction).
   The DICOMs are uncompressed, so the old 2 h was ~63% per-file read latency, not compute.
3. **One model replica per T4** (the Kaggle box exposes two): study i -> GPU i mod 2, start-up cross-GPU self-check,
   retry-once on the other GPU, retirement after 3 consecutive failures.
4. Safety: `cv2.setNumThreads(1)` set in the PARENT before fork (setting it inside a forked child deadlocks once
   the parent's cv2 pool is live — reproduced); a 300 s watchdog that terminates a hung pool and finishes in-process.

## builder/
`build_eff3.py OUT SLUG CKPT ARCH SHA K BUNDLE W PF` derives the kernel from `base_kernel/rsna_dread_D96.py` by
asserted, count-checked substitutions. The scored v3 used:
`kaggle_kernel_dreadEff3 eff-d96-k43-2gpu raptor_ft_rsnaD96F42_swa.pt coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k 902ec763...a0ad85 43 pareru806/kaggle-bundle-dread-d96 6 8`.
Weights: the same d96 s42 checkpoint as the accuracy model (`../public_LB_model/coatnet_d96/`).
