import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(data):
    data['Category General'] = data['Category General'].fillna("[]")
    data['Category General'] = data['Category General'].apply(
        lambda x: x.strip("[]").replace("'", "").split(", ") if isinstance(x, str) else []
    )
    data['Category_General_Text'] = data['Category General'].apply(
        lambda x: " ".join(x) if isinstance(x, list) else ""
    )
    return data
