import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

import time

import numpy as np
import pandas as pd

from cl import ConditionalLogit
from dataset import Dataset
from sampling import sample, Winter2022, Winter2023, Winter2024, MenWaregroup
from encoding import encode_seasonality
from data import load
from season_range import SeasonRange
from coeffcients import build_coefficients
from evaluate import evaluate

params = dict(season=SeasonRange(seasons=(Winter2024, Winter2023, Winter2022)))

df = sample(
    pd.read_csv("data/sales.csv", parse_dates=["date"]),
    store=(1, 2, 3, 4),
    season=params["season"],
    waregroup=MenWaregroup,
    n_items=None,
    n_choices=25000,
    random_seed=42,
    key="category",
)

df = encode_seasonality(df)

df = pd.get_dummies(df, columns=["store"], dtype=np.int32)

df.columns = df.columns.str.upper()

df = df.drop(columns=["QUANTITY", "SALE_TYPE", "SKU", "DATE"])

df["SALE_AMOUNT"] = df["SALE_AMOUNT"].astype(np.float32)
df["CATEGORY"] = df["CATEGORY"].astype(np.int32)


items_feature_groups = [
    {"name": "sale_amount", "columns": ["SALE_AMOUNT"]},
]

shared_feature_groups = [
    {"name": "season_context", "columns": ["START_SEASON", "MID_SEASON", "END_SEASON"]},
    {
        "name": "store",
        "columns": [f"STORE_{i}" for i in range(1, 5)],
    },
]

ds = load(df, items_feature_groups, shared_feature_groups, key="CATEGORY")

ds.summary()

coefficients = build_coefficients(
    shared_feature_groups=shared_feature_groups,
    items_feature_groups=items_feature_groups,
    shared_type="item",
    items_type="constant",
)

print(f"\nCoefficients: {coefficients}")


model = ConditionalLogit(coefficients=coefficients, optimizer="lbfgs")
# model = ConditionalLogit(
#     coefficients=coefficients,
#     optimizer="adam",
#     lr=0.01,
#     epochs=200,
#     batch_size=64,
# )

start = time.perf_counter()
history = model.fit(ds, get_report=False, verbose=2)
end = time.perf_counter()

print(f"Training finished in: {end-start:.2f}s.")

evaluate(model, ds)
