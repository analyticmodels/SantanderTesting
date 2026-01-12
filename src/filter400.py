import pandas as pd

df = pd.read_csv("data/hallucinationQuestions_responses_all.csv")

# Count rows with "Error 400" (handle NaN values)
error_mask = df["answer"].fillna("").str.startswith("Error 400")
print(f"Rows with Error 400: {error_mask.sum()}")

# Filter out rows that start with "Error 400"
dfResp = df[~error_mask].copy()

dfResp.to_csv("data\hallucinationQuestions_no400_all.csv")
print(dfResp.head())
print(f"Filtered row count: {len(dfResp)}")
