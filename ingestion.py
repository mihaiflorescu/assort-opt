import pandas as pd
from data import load


sales = pd.read_csv("data/sales.csv", parse_dates=["date"]).dropna()

assert sales.quantity.eq(-1).all()  # Ensure quantity is always -1
assert sales.sale_amount.gt(0).all  # Ensure sale price is always positive
assert sales.colour.between(0, 9).all()  # Ensure colour is always between 0 and 9

sales.info()

variants = pd.read_csv("variants_general.csv")

variants.info()

sales_enriched = sales.merge(variants, on=["colour", "category"], how="inner")

sales_enriched.drop(columns=["sale_type", "SKU", "quantity"], inplace=True)

sales_enriched.rename(columns={"general_id": "ID"}, inplace=True)

sales_enriched.info()


sales_enriched.to_csv("men_variants.csv", index=False)

sales_enriched.category.value_counts()


ds = load(sales_enriched, items_feature_groups=["sale_amount"])
