# generate_sample_data.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

random.seed(42)
np.random.seed(42)

products = [
    ("Laptop",       "Electronics",  800, 1200),
    ("Phone",        "Electronics",  400,  900),
    ("Mouse",        "Accessories",   20,   60),
    ("Keyboard",     "Accessories",   40,  120),
    ("Monitor",      "Electronics",  200,  500),
    ("Desk",         "Furniture",    150,  400),
    ("Chair",        "Furniture",    100,  350),
    ("Headphones",   "Accessories",   50,  200),
    ("Webcam",       "Electronics",   60,  150),
    ("Notebook",     "Stationery",     5,   20),
]

rows = []
start_date = datetime(2023, 1, 1)

for i in range(500):
    customer_id  = f"C{random.randint(1, 80):03d}"
    product      = random.choice(products)
    name, cat, low, high = product
    date         = start_date + timedelta(days=random.randint(0, 365))
    quantity     = random.randint(1, 3)
    price        = round(random.uniform(low, high), 2)
    rows.append({
        "customer_id":   customer_id,
        "transaction_id": f"T{i+1:04d}",
        "product_name":  name,
        "category":      cat,
        "quantity":      quantity,
        "price":         price,
        "date":          date.strftime("%Y-%m-%d"),
    })

df = pd.DataFrame(rows)
df.to_csv("sample_data.csv", index=False)
print(f"Generated {len(df)} rows")
print(df.head())
