# Olist source data

Download `olistbr/brazilian-ecommerce` from Kaggle (CC BY-NC-SA 4.0), unzip it, and copy these files into `data/raw/`:

- `olist_orders_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_products_dataset.csv`
- `product_category_name_translation.csv`

The raw data are intentionally ignored by git. A Kaggle API user can run `kaggle datasets download -d olistbr/brazilian-ecommerce -p data/raw --unzip` from the repository root.
