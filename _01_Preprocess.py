import numpy as np
import torch
import pandas as pd
from pathlib import Path
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split

dataset = load_dataset("bevaya/FinQA")

train_df = dataset["train"].to_pandas()
val_df = dataset["validation"].to_pandas()
test_df = dataset["test"].to_pandas()

def clean_data(df):
    duplicates = df["id"].duplicated().sum()
    df = df.drop_duplicates(subset="id")

    if df["question"].isnull().any() or df["answer"].isnull().any():
        df = df.dropna(subset=["question", "answer"])

    return df

train_df = clean_data(train_df)
val_df = clean_data(val_df)
test_df = clean_data(test_df)

def format_data(value):
    if value is None:
        return ""

    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) > 0 and isinstance(value[0], (list, tuple, np.ndarray)):
            return "\n".join(" | ".join("" if cell is None else str(cell) for cell in row) for row in value)
        return "\n".join(str(item).replace("\n", " ") for item in value)

    if pd.isna(value):
        return ""
    return str(value).replace("\n", " ")

def create_prompt(row):
    return f"""You are a financial analysis assistant.
    Context: {format_data(row['pre_text'])}
    Financial table: {format_data(row['table'])}
    Additional context: {format_data(row['post_text'])}
    Question: {row['question']}
    Answer:
    """

def prepare_dataset(df):
    records = []

    for _, row in df.iterrows():
        records.append({"prompt": [{"role": "system", "content": ( "You are a financial analysis assistant. ""Answer accurately using the provided information.")}, {"role": "user", "content": create_prompt(row)}], "completion": [{"role": "assistant", "content": str(row["answer"]).strip()}]})
    return Dataset.from_list(records)

train_dataset = prepare_dataset(train_df)
val_dataset = prepare_dataset(val_df)
test_dataset = prepare_dataset(test_df)

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)
train_dataset.save_to_disk(str(output_dir / "train"))
val_dataset.save_to_disk(str(output_dir / "validation"))
test_dataset.save_to_disk(str(output_dir / "test"))