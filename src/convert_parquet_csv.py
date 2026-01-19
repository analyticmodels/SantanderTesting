import pandas as pd

parquet_file="data/hallucinationQuestions_meta-llama_llama-3-3-70b-instruct.parquet"

df = pd.read_parquet(parquet_file)
csv_file = f"{parquet_file[:-8]}.csv"
print (csv_file)
df.to_csv(csv_file)