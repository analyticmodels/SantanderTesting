import pandas as pd
from pathlib import Path

# Read sentences from the file
data_file = Path('../data/questions.txt')
with open(data_file, 'r', encoding='utf-8') as f:
    sentences = [line.strip() for line in f if line.strip()]

# Create DataFrame
df = pd.DataFrame({'question': sentences})

# Remove duplicates
df_dedup = df.drop_duplicates(subset=['question'], keep='first')

# Save as parquet using fastparquet engine to avoid PyArrow conflicts
output_file = Path('../data/baseQuestions.parquet')
df_dedup.to_parquet(output_file, engine='fastparquet', index=False)

print(f"Total sentences: {len(df)}")
print(f"Deduplicated sentences: {len(df_dedup)}")
print(f"Duplicates removed: {len(df) - len(df_dedup)}")
print(f"Saved to: {output_file}")