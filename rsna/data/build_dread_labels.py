#!/usr/bin/env python
"""labels_llm_public -> the reference training script's label format, DENSE.
The author's labels (labels_llm_soft.parquet) were never published; labels_llm_public is the public
CC0 set in the SAME FORMAT (soft probabilities defined on every cell). The reference trains with NO
cell mask, so our w_ columns are deliberately DROPPED -- that is the 'dense unmasked BCE' part of the
bundle, never tested before (every earlier public-label run kept our mask)."""
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
LAB = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA",
       "Effusion","Synovitis","Baker's","Contusion","Fracture"]      # the reference's LAB, in order
d = pd.read_parquet(ROOT/"artifacts/labels/labels_llm_public.parquet")
out = pd.DataFrame({"StudyInstanceUID": d.index.astype(str)})
for l in LAB: out[l] = d[f"y_{l}"].values.astype(np.float32)
assert not out[LAB].isna().any().any(), "dense labels must be defined on every cell"
p = ROOT/"dread/labels_public_dense.parquet"; out.to_parquet(p, index=False)
print(f"wrote {p}  {out.shape} | soft cells {np.mean((out[LAB].values>0.01)&(out[LAB].values<0.99))*100:.1f}%"
      f" | NaN {out[LAB].isna().values.mean()*100:.1f}%")
