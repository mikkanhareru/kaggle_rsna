# preprocessing — corpora, caches and labels, in build order

All builders are modules of `../rsna/` (run from `key_idea/` as `python -m rsna...`).

## 1. Labels (shared by every model)
| Step | Command | Output |
|---|---|---|
| Public CC0 report labels -> our label format | `python -m rsna.labels.build_public` | public label tables |
| Dense soft labels for the CoAtNet trainer | `python -m rsna.data.build_dread_labels` | reads `artifacts/labels/labels_llm_public.parquet` (public CC0 labels, also the trainer's `--rank_mask`), writes `dread/labels_public_dense.parquet` (the training target) |
| Our own LLM extraction (SSL member's `--policy fused` target) | `rsna.labels.extract` -> `rsna.labels.consensus` -> `rsna.labels.build_labels` | fused label tables |

## 2. CoAtNet corpora (the "Dread" volumes)
| Step | Command | Output |
|---|---|---|
| 64-slice corpus (R / RGR) | `python -m rsna.data.build_dread_corpus --workers 32` | `dread/train/all_vols.npy` (4407, 64, 336, 336) + masks + ids |
| **DENSE96** corpus (d96, used by both final models) | `python -m rsna.data.build_dread_corpus_d96 --workers 32` | `all_vols_d96.npy` (4407, 96, 384, 384), ~62 GB; launch with `setsid nohup` |

Both builders extract the preprocessing from `reference/dread_0924_inference.py` and refuse to run unless its
sha256 is `859aae12...`; DENSE96 applies exactly two counted substitutions (SLOTS x1.5, IMG 384).

## 3. SSL-member caches
| Step | Command | Output |
|---|---|---|
| Series header audit | `python -m rsna.data.audit_slots --split train --dump` | header table |
| Slot cache (fine-tune input) | `python -m rsna.data.build_slot_cache` | `cache_slots_336_130mm_12sl_nb/` |
| SSL plane cache (MAE input; refuses fold-4 studies) | `python -m rsna.data.build_ssl_cache` | `ssl_cache_336_130mm/` |
