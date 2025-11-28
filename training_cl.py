import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

import time

import numpy as np
import pandas as pd

from cl import ConditionalLogit
from dataset import Dataset
from sampling import sample, Winter2023

df = sample(
    pd.read_csv("data/men_variants.csv"),
    season=Winter2023,
    n_items=10,
    n_choices=10000,
    random_seed=42,
)

periods = {
    9: "START_SEASON",
    10: "START_SEASON",
    11: "MID_SEASON",
    12: "MID_SEASON",
    1: "END_SEASON",
    2: "END_SEASON",
}

df = df.assign(
    date=pd.to_datetime(df["date"]),
    period=lambda d: d["date"].dt.month.map(periods),
)

df = df.join(pd.get_dummies(df["period"]).astype("int32"))

df = pd.get_dummies(df, columns=["colour"], dtype=np.int32)

df = df.drop(columns=["date", "period"])

df.columns = df.columns.str.upper()

df["SALE_AMOUNT"] = df["SALE_AMOUNT"].astype(np.float32)
df["ID"] = df["ID"].astype(np.int32)
df["CATEGORY"] = df["CATEGORY"].astype(np.int32)


items_feature_groups = [
    {"name": "sale_amount", "columns": ["SALE_AMOUNT"]},
    {
        "name": "colour",
        "columns": [col for col in df.columns if col.startswith("COLOUR_")],
    },
]

shared_feature_groups = [
    {"name": "season_context", "columns": ["START_SEASON", "MID_SEASON", "END_SEASON"]}
]

unique_skus = np.sort(df.ID.unique())
id_to_index = {product_id: i for i, product_id in enumerate(unique_skus)}
n_items = len(id_to_index)

shared_features_by_choice_tuple = None
shared_features_names_tuple = None

if shared_feature_groups is not None:
    shared_features_lists = tuple([] for _ in shared_feature_groups)
    shared_features_names_list = []

    for group_idx, group in enumerate(shared_feature_groups):
        columns = group["columns"]
        shared_features_names_list.append(columns)

        for i, row in df.iterrows():
            shared_features_lists[group_idx].append(row[columns].values)

    shared_features_by_choice_tuple = tuple(
        np.array(group_list) for group_list in shared_features_lists
    )
    shared_features_names_tuple = tuple(shared_features_names_list)

items_features_by_choice_tuple = None
items_features_names_tuple = None

if items_feature_groups is not None:
    items_features_list = []
    items_features_names_list = []

    for group in items_feature_groups:
        columns = group["columns"]
        n_features = len(columns)

        feature_matrix = np.zeros((n_items, n_features))

        for sku, idx in id_to_index.items():
            sku_data = df.loc[df.ID == sku, columns]
            if len(sku_data) > 0:
                feature_matrix[idx, :] = sku_data.iloc[0].values

        items_features_list.append(feature_matrix)
        items_features_names_list.append(columns)

    items_features_by_choice_groups = tuple([] for _ in items_feature_groups)

    for i, row in df.iterrows():
        for group_idx in range(len(items_feature_groups)):
            items_features_by_choice_groups[group_idx].append(
                items_features_list[group_idx]
            )

    items_features_by_choice_tuple = tuple(
        np.array(group_list) for group_list in items_features_by_choice_groups
    )
    items_features_names_tuple = tuple(items_features_names_list)

choices = []
for i, row in df.iterrows():
    item_index = id_to_index[row.ID]
    choices.append(item_index)

ds = Dataset(
    shared_features_by_choice=shared_features_by_choice_tuple,
    shared_features_by_choice_names=shared_features_names_tuple,
    items_features_by_choice=items_features_by_choice_tuple,
    items_features_by_choice_names=items_features_names_tuple,
    choices=choices,
)

ds.summary()

coefficients = {
    "START_SEASON": "constant",
    "MID_SEASON": "constant",
    "END_SEASON": "constant",
    "SALE_AMOUNT": "constant",
}

for col in df.columns:
    if col.startswith("COLOUR_"):
        coefficients.update({col: "constant"})

model = ConditionalLogit(coefficients=coefficients, optimizer="lbfgs")

start = time.perf_counter()
history = model.fit(ds, get_report=True, verbose=2)
end = time.perf_counter()

print(f"Number of items: {ds.base_num_items}")
print(f"Number of choices: {len(ds)}")
print(f"Number of parameters: {sum(w.shape[1] for w in model.trainable_weights)}")

print("The average neg-loglikelihood is:", model.evaluate(ds).numpy())
print("The total neg-loglikelihood is:", model.evaluate(ds).numpy() * len(ds))

probas = model.predict_probas(ds)
print("Min probability: ", np.min(probas))
print("Max probability: ", np.max(probas))

accuracy = np.mean(np.argmax(probas, axis=1) == ds.choices)

print(f"Top-1 Accuracy: {accuracy:.2%}")

print(f"Training finished in: {end-start:.2f}s.")
