import pandas as pd

csv_file="data/agentAssistBaseQuestions.csv"

df = pd.read_csv(csv_file)
parquet_file = f"{csv_file[:-4]}.parquet"
print (parquet_file)
df.to_parquet(parquet_file)