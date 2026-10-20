# Pilot (default):  ~120 language-stratified reports, greedy, 1 sample.
# Full run:         --all --n-samples 3 --temperature 0.3 --tag full

from __future__ import annotations
import argparse, json, re, sys
from collections import Counter
from pathlib import Path
import pandas as pd
import py3langid as langid
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rsna.labels.schema import ReportExtraction, COLUMN_MAP
from rsna.labels.prompt import build_messages
from rsna.labels.splits import gold_dev_uids
from rsna.labels.consensus import consensus

DATA = ROOT / "data"
OUT = ROOT / "artifacts" / "labels"

# 1. language 
_BCS = {"bs": "hr/bs", "hr": "hr/bs", "sr": "hr/bs"}

def guess_lang(s: str) -> str:
    code = langid.classify(str(s))[0]
    return _BCS.get(code, code)

# 2. selection 
def load_reports() -> pd.DataFrame:
    df = pd.read_csv(DATA / "train.csv")
    labels = [c for c in df.columns if c not in ("StudyInstanceUID", "Report")]
    df["is_gold"] = df[labels].notna().all(axis=1)
    df["lang_guess"] = df.Report.map(guess_lang) 
    return df

def select(df: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    dev  = df[df.StudyInstanceUID.isin(gold_dev_uids(df))]
    pool = df[~df.is_gold]
    per  = max(1, (limit - len(dev)) // pool.lang_guess.nunique())
    extra = pd.concat([g.sample(min(len(g), per), random_state=seed)
                       for _, g in pool.groupby("lang_guess")])
    return pd.concat([dev, extra]).reset_index(drop=True)

# 4. main 
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-32B-AWQ")
    ap.add_argument("--all", action="store_true", help="run all 4,407 reports")
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--n-samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    df  = load_reports()
    sub = df if a.all else select(df, a.limit, a.seed)
    print(f"selected {len(sub)} reports | {sub.lang_guess.value_counts().to_dict()}")
    
    llm = LLM(
        model=a.model,
        quantization="awq",
        max_model_len=a.max_model_len,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,      # the ~1.1k-token SYSTEM block is reused every call
        seed=a.seed,
    )
    sp = SamplingParams(
        n=a.n_samples, temperature=a.temperature, max_tokens=2048, seed=a.seed,
        structured_outputs=StructuredOutputsParams(
            json=ReportExtraction.model_json_schema()),
    )
    outs = llm.chat(
        [build_messages(r) for r in sub.Report], sp,
        chat_template_kwargs={"enable_thinking": False},   # Qwen3 CoT vs guided JSON
    )

    rows, raw, n_bad = [], [], 0
    for (_, meta), o in zip(sub.iterrows(), outs):
        texts, parsed = [c.text for c in o.outputs], []
        for t in texts:
            try: parsed.append(ReportExtraction.model_validate_json(t))
            except Exception: pass
        raw.append(dict(StudyInstanceUID=meta.StudyInstanceUID,
                        n_ok=len(parsed), raw=json.dumps(texts)))
        if not parsed:
            n_bad += 1
            continue
        for key, v in consensus(parsed).items():
            rows.append(dict(
                StudyInstanceUID=meta.StudyInstanceUID,
                column=COLUMN_MAP[key], finding=key,
                lang_guess=meta.lang_guess,
                lang_llm=Counter(p.language for p in parsed).most_common(1)[0][0],
                is_gold=bool(meta.is_gold), **v))

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(OUT / f"findings_{a.tag}.parquet", index=False)
    pd.DataFrame(raw ).to_parquet(OUT / f"raw_{a.tag}.parquet",      index=False)
    print(f"\nparsed {len(sub) - n_bad}/{len(sub)} reports -> "
          f"{len(rows)} finding-rows in {OUT}/findings_{a.tag}.parquet")

if __name__ == "__main__":
    main()