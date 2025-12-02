import numpy as np
import pandas as pd
from typing import NamedTuple


MenWaregroup = (41, 42, 43, 45, 46, 47, 48, 74, 84)


class Season(NamedTuple):
    Start: str
    End: str


Summer2024 = Season("2024-03-01", "2024-08-31")
Summer2023 = Season("2023-03-01", "2023-08-31")
Summer2022 = Season("2022-03-01", "2022-08-31")

Winter2024 = Season("2024-09-01", "2025-02-28")
Winter2023 = Season("2023-09-01", "2024-02-29")
Winter2022 = Season("2022-09-01", "2023-02-28")


def sample(
    _df: pd.DataFrame,
    key="SKU",
    season=None,
    store=None,
    waregroup=None,
    n_items=10,
    n_choices=100,
    random_seed=42,
) -> pd.DataFrame:
    df = _df.copy()

    if season is not None:
        mask = season.create_mask(df, date_column="date")
        df = df[mask]

    if store is not None:
        df = df[df.store.isin(store)]

    if waregroup is not None:
        df = df[df.category.astype(str).str[:2].astype(int).isin(waregroup)]

    print(f"Max Number of choices: {len(df)}")

    if n_choices is None:
        top_n_ids = df[key].value_counts().head(n_items).index
        choices = df[df[key].isin(top_n_ids)]

        return choices.sample(n=n_choices, random_state=random_seed)

    return df.sample(n=n_choices, random_state=random_seed)
