import numpy as np
import pandas as pd


def get_column_dtypes(df: pd.DataFrame) -> dict:
    return df.dtypes.apply(lambda x: x.name).to_dict()


def encode_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    season_map = {
        "START_SEASON": [9, 10],
        "MID_SEASON": [11, 12],
        "END_SEASON": [1, 2],
    }
    periods = {m: s for s, months in season_map.items() for m in months}

    df = df.assign(period=lambda d: d.date.dt.month.map(periods))

    df = df.join(pd.get_dummies(df.period, dtype=np.int32)).drop(columns=["period"])

    return df
