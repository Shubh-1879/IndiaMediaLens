import pandas as pd
from sqlalchemy import create_engine

# 1. Load your NEW batches
df_new = pd.read_csv(r"C:\Users\shubh\Downloads\MSc 3rd Sem\IndiaMediaLens\Processed Data\Zero Shot\Laptop Batches\combined_7-12.csv") 

# 2. Connect to the database
engine = create_engine("mysql+pymysql://root:#Shubhpass198#@localhost:3306/laptop_absa_project")

# 3. Push to SQL using 'append' instead of 'replace'
# This safely drops the new rows at the bottom of the existing table
df_new.to_sql(name='gemini_results', con=engine, if_exists='append', index=False)

print("Batches 7-12 successfully added!")