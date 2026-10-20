# public_LB_model — d96-ssl, public LB 0.936

`final = 0.4 * rank(d96 s42) + 0.4 * rank(d96 s43) + 0.2 * rank(SSL s4mae)`, per-column percentile ranks.
Weights 0.4/0.4/0.2 were registered on 2026-09-29 (`../docs/preregistered/2026-09-29_SSL_member.md`) and never tuned on the LB.

## coatnet_d96/ — the two CoAtNet-2 arms
- `knee_train_d96.py`: standalone trainer (timm + torch; no `rsna` import). `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`,
  attention-MIL head over 2.5D windows, DropPath 0.1, rank loss 0.5 / tau 1.0, SWA. Reads the DENSE96 corpus via
  `RSNA_CORPUS` / `RSNA_MASKS`.
- `run_d96.sh 42` and `run_d96.sh 43`: full-data runs (`--folds 0`), 16 epochs, ~2 h each on an H100;
  checks corpus shape / ids / GPU before launching. Outputs `raptor_ft_rsnaD96F{42,43}_swa.pt`.

## ssl_member/ — s4mae (ViT-S/14, 5 folds)
1. `run_ssl1.sh`: continued MAE pretraining of DINOv2 ViT-S/14 on the SSL plane cache (`rsna.train.pretrain_mae`);
   the shipped encoder is step 150,000 (`mae_ssl1_s150000.pt`, sha `353becb85cf4053c`).
2. `run_s5_folds.sh`: the ep50nb recipe with `--encoder-init` = that MAE checkpoint and encoder LR x0.081, all folds
   (`rsna.train.loop` + `rsna.train.predict`). `run_ep50nb.sh` is the base recipe it copies.

## kernel/ — the Kaggle submission
`rsna_dread_D96_ssl.py`: phase 1 scores the SSL member (its own slot decode, code shipped inside the s4mae bundle),
phase 2 scores the two CoAtNet arms (prefetch thread, K_EVAL 86, sha256-checked weights), then blends.

## bundles/ — private Kaggle datasets the kernel mounts (metadata only, no weights here)
`dread_d96` (d96 s42), `dread_d96b` (d96 s43), `slot_s4mae` (5 folds + its `rsna` code + provenance).
