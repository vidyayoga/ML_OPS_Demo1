from sklearn.datasets import load_breast_cancer
import pandas as pd


# Load dataset
dataset = load_breast_cancer()


# Convert to DataFrame
df = pd.DataFrame(
    dataset.data,
    columns=dataset.feature_names
)


# Add target column
df["target"] = dataset.target


# Save CSV
df.to_csv(
    "data/breast_cancer.csv",
    index=False
)


print("Dataset downloaded and saved successfully!")
print("Shape:", df.shape)
print(df.head())