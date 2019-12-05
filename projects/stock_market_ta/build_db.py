import pandas as pd
import os
from sqlalchemy import create_engine

HOST = 'localhost'
USER = 'postgres'
PASS = 'password'
db = 'stock_market'
conn_string = f"host={HOST} port=5432 dbname={db} user={USER} password={PASS}"
engine = create_engine(f'postgresql://{USER}:{PASS}@{HOST}:5432/{db}')

for file in os.listdir('data/daily'):
    print(file)
    test = pd.read_csv(f'data/daily/{file}')
    test['ticker'] = file.split('.')[0]
    test.to_sql('market_data',if_exists='append',con=engine,index=False)