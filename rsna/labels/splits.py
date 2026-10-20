"""Gold dev/locked split.

Deliberately dependency-free (pandas only) so analysis scripts can import it
without pulling in vLLM or py3langid, which live only in the rsna_llm env.
"""
from pathlib import Path

import pandas as pd

GOLD_DEV_N = 12


def load_gold_flags(data_dir: Path) -> pd.DataFrame:
    """train.csv with an is_gold boolean column added."""
    tr = pd.read_csv(data_dir / "train.csv")
    labels = [c for c in tr.columns if c not in ("StudyInstanceUID", "Report")]
    tr["is_gold"] = tr[labels].notna().all(axis=1)
    return tr


def gold_dev_uids(df: pd.DataFrame) -> set[str]:
    """The 12 gold studies visible while developing the prompt.

    Includes the first 3, which were already read during the data survey.
    The other 46 stay locked. `df` needs StudyInstanceUID and a boolean is_gold.
    """
    gold = df[df.is_gold]
    return set(gold.head(3).StudyInstanceUID) | set(
        gold.iloc[3:].sample(GOLD_DEV_N - 3, random_state=0).StudyInstanceUID
    )
