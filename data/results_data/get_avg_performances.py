import pandas as pd
import os

fname_list = [os.path.join(*x) for x in 
  [
    ["ONESP", "high_beta", "comparison_100.csv.xz"],
    ["1FISHERY", "DEFAULT", "comparison_300.csv.xz"],
    ["2FISHERY", "DEFAULT", "comparison_300.csv.xz"],
    ["2FISHERY", "RXDRIFT", "comparison_300.csv.xz"],
  ]
]

print(fname_list)

for fname in fname_list:
  df = pd.read_csv(fname)
  print(
    "\n###########################################\n", 
    fname, 
    "\n###########################################\n",
  )
  for strategy in ['CMort', 'CEsc', 'PPO', 'PPO+GP']:
    print(f"strategy = {strategy}")
    df_strat = df.loc[df.strategy == strategy]
    avg = df_strat.agg({'reward':'mean'})
    df_maxt = df_strat.loc[df.t == 199]
    frac = len(df_maxt.index)/len(df_strat)
    print(avg)
    print("fraction max t=199:", frac, "\n")
  
