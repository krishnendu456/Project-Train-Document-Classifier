import pandas as pd
import os

# Create data folder automatically
os.makedirs("data", exist_ok=True)

url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"

df = pd.read_csv(
    url,
    sep='\t',
    header=None,
    names=['label', 'text']
)

df.to_csv("data/spam_dataset.csv", index=False)

print("Dataset saved successfully!")
print(df.head())