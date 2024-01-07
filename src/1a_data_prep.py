import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def generate_df(raw_file_loc: str):
  df_gas_comp = pd.read_excel(raw_file_loc, sheet_name="Gas Comp")\
              .fillna(0)\
              .rename(columns={'QNG-S1b': 'QNG1', 'QNG-S2c': 'QNG2', 'QNG-S3d': 'QNG3', 'QNG-S4e': 'QNG4', 'QNG-S5f': 'QNG5'})
  df_rho_ng_1 = pd.read_excel(raw_file_loc, sheet_name="QNG-S1", index_col=0, header=1)
  df_rho_ng_2 = pd.read_excel(raw_file_loc, sheet_name="QNG-S2", index_col=0, header=1)
  df_rho_ng_3 = pd.read_excel(raw_file_loc, sheet_name="QNG-S3", index_col=0, header=1)
  df_rho_ng_4 = pd.read_excel(raw_file_loc, sheet_name="QNG-S4", index_col=0, header=1)
  df_rho_ng_5 = pd.read_excel(raw_file_loc, sheet_name="QNG-S5", index_col=0, header=1)

  dfs = [df_rho_ng_1, df_rho_ng_2, df_rho_ng_3, df_rho_ng_4, df_rho_ng_5]
  processed_dfs = [] # prepare list for all melted dfs

  # iterate to all df_rho_ng (1-5)
  for i, df in enumerate(dfs, start=1):
    # pandas df melt
    melted_df = df\
                  .reset_index()\
                  .melt(id_vars='Mpa \ Kelvin', var_name='Temperature', value_name='Density')\
                  .assign(NG_TYPE=i)
    
    # map df_gas_comp data to the melted df
    for _, row in df_gas_comp.iterrows():
      melted_df[row['Component']] = row[f'QNG{i}']
    
    # append the melted df to processed df list
    processed_dfs.append(melted_df)

  # join the all the modified melted dfs
  grand_df = pd.concat(processed_dfs, axis=0)\
              .dropna(subset=['Density'])\
              .rename(columns={'Mpa \ Kelvin': 'Pressure'})\
              .reset_index(drop=True)

  # convert Temperature col to int64 datatype
  grand_df['Temperature'] = grand_df['Temperature'].astype('int64')

  return grand_df

if __name__=='__main__':
  raw_file_loc = "dataset/Natural gas density data.xlsx"
  df = generate_df(raw_file_loc)

  print(df)

  save_loc = "dataset/ng_density_all.csv"
  df.to_csv(save_loc, index=False)
  print(f"data saved to {os.path.abspath(save_loc)}")