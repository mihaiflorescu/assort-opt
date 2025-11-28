import pandas as pd
from typing import NamedTuple


class Season(NamedTuple):
    Start: str
    End: str


Winter2024 = Season("2024-09-01", "2025-02-28")
Winter2023 = Season("2023-09-01", "2024-02-29")
Winter2022 = Season("2022-09-01", "2023-02-28")


def sample(
    _df: pd.DataFrame, season: Season, n_items=10, n_choices=100, random_seed=42
) -> pd.DataFrame:
    df = _df.copy()
    df.date = pd.to_datetime(df.date)
    df = df[(df.date >= season.Start) & (df.date <= season.End)]
    top_n_ids = df.ID.value_counts().head(n_items).index
    choices = df[df["ID"].isin(top_n_ids)]
    return choices.sample(n=n_choices, random_state=random_seed)
