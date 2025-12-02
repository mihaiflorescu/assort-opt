from typing import List, Dict


def build_coefficients(
    shared_feature_groups: List[Dict[str, any]],
    items_feature_groups: List[Dict[str, any]],
    shared_type: str = "constant",
    items_type: str = "constant",
) -> Dict[str, str]:
    """Build a coefficients dictionary from feature groups.

    Args:
        shared_feature_groups: List of dicts with 'name' and 'columns' keys for shared features
        items_feature_groups: List of dicts with 'name' and 'columns' keys for items features
        shared_type: Coefficient type for shared features (default: "constant")
        items_type: Coefficient type for items features (default: "constant")

    Returns:
        Dictionary mapping column names to coefficient types
    """
    coefficients = {}

    # Extract columns from shared feature groups
    for group in shared_feature_groups:
        for col in group["columns"]:
            coefficients[col] = shared_type

    # Extract columns from items feature groups
    for group in items_feature_groups:
        for col in group["columns"]:
            coefficients[col] = items_type

    return coefficients
