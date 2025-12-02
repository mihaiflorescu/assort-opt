import numpy as np

from dataset import Dataset


def load(df, items_feature_groups, shared_feature_groups=None, key="SKU"):
    unique_skus = np.sort(df[key].unique())
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

            for _, row in df.iterrows():
                shared_features_lists[group_idx].append(row[columns].values)

        shared_features_by_choice_tuple = tuple(
            np.array(group_list) for group_list in shared_features_lists
        )
        shared_features_names_tuple = tuple(shared_features_names_list)

    items_features_by_choice_tuple = None
    items_features_names_tuple = None

    items_features_list = []
    items_features_names_list = []

    for group in items_feature_groups:
        columns = group["columns"]
        n_features = len(columns)

        feature_matrix = np.zeros((n_items, n_features))

        for sku, idx in id_to_index.items():
            sku_data = df.loc[df[key] == sku, columns]
            if len(sku_data) > 0:
                feature_matrix[idx, :] = sku_data.iloc[0].values

        items_features_list.append(feature_matrix)
        items_features_names_list.append(columns)

    items_features_by_choice_groups = tuple([] for _ in items_feature_groups)

    for _, row in df.iterrows():
        for group_idx in range(len(items_feature_groups)):
            items_features_by_choice_groups[group_idx].append(
                items_features_list[group_idx]
            )

    items_features_by_choice_tuple = tuple(
        np.array(group_list) for group_list in items_features_by_choice_groups
    )
    items_features_names_tuple = tuple(items_features_names_list)

    choices = []
    for _, row in df.iterrows():
        item_index = id_to_index[row[key]]
        choices.append(item_index)

    ds = Dataset(
        shared_features_by_choice=shared_features_by_choice_tuple,
        shared_features_by_choice_names=shared_features_names_tuple,
        items_features_by_choice=items_features_by_choice_tuple,
        items_features_by_choice_names=items_features_names_tuple,
        choices=choices,
    )

    return ds
