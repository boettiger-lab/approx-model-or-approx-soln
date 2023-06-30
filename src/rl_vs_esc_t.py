import numpy as np
import os
import pandas as pd
from plotnine import (
  ggplot, geom_point, aes, geom_line, facet_wrap, geom_path, geom_bar, geom_histogram
)
import matplotlib.pyplot as plt

MAXT = 200
BINSIZE = 50
BINSTARTS = np.linspace(
  0, MAXT-BINSIZE, (MAXT-BINSIZE)//BINSIZE+1
) # // is integer division (leave out remainder)

DATACODE = "RXDRIFT"
ENVCODE = "2FISHERY"
DATAPATH = os.path.join("../data/results_data", ENVCODE, DATACODE)

def avg_action_difference_at_bin(binstart, df1, df2):
	df1 = df1.loc[
		(df1.t >= binstart) & (df1.t<binstart+BINSIZE) 
	]
	df2 = df2.loc[
		(df2.t >= binstart) & (df2.t<binstart+BINSIZE) 
	]
	return (
	  (df2["act_x"] - df1["act_x"]).mean(), 
	  (df2["act_x"] - df1["act_x"]).std(),
	  (df2["act_y"] - df1["act_y"]).mean(),
	  (df2["act_y"] - df1["act_y"]).std(),
	 )

def avg_action_difference(df1, df2):
	return pd.DataFrame(
	  [[
			binstart, 
			*avg_action_difference_at_bin(binstart, df1, df2), 
			f"{int(binstart)}-{int(binstart+BINSIZE)}",
		] 
		for binstart in BINSTARTS ],
  	columns = [
  	  "bin_start", 
  	  "avg_act_x_diff", 
  	  "std_act_x_diff", 
  	  "avg_act_y_diff", 
  	  "std_act_y_diff", 
  	  "bin_range",
  	]
	)
	
esc_df = pd.read_csv(os.path.join(DATAPATH, "esc.csv.xz"))
ppo_df = pd.read_csv(os.path.join(DATAPATH, "ppo201.csv.xz"))

#print(avg_action_difference(esc_df, ppo_df)[["bin_range", "avg_act_x_diff", "avg_act_y_diff"]].head(20))
result_df = avg_action_difference(esc_df, ppo_df)


## PLOTS

# color
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
color_gradient = cm.get_cmap('Pastel1', 12)
color_choices = color_gradient([0,2])
newcmp = ListedColormap(color_choices)


## PLOT 1

fig = plt.figure()
plt.errorbar(
  result_df.bin_range, result_df.avg_act_x_diff, 
  yerr=result_df.std_act_x_diff, label="(PPO - CEsc) X action", fmt="o--",
  capsize=5,
  color = "lightcoral",
)
plt.errorbar(
  result_df.bin_range, result_df.avg_act_y_diff, 
  yerr=result_df.std_act_y_diff, label="(PPO - CEsc) Y action", fmt="o--",
  capsize=5,
  color = "cornflowerblue",
)
plt.xlabel("Time Bin")
plt.ylabel("Difference in action")
plt.legend(loc="best")
plt.yticks([-0.1, 0, 0.1, 0.2, 0.3], minor=False)
plt.grid(axis="y")
plt.savefig(fname=f"{DATAPATH}/difference.png")
plt.close()


## PLOT 2
# https://stackoverflow.com/questions/63866002/how-to-add-error-bars-to-a-grouped-bar-plot

# error bar bounds
result_df["x_low"] = result_df["avg_act_x_diff"] - result_df["std_act_x_diff"]
result_df["x_high"] = result_df["avg_act_x_diff"] + result_df["std_act_x_diff"]
result_df["y_low"] = result_df["avg_act_y_diff"] - result_df["std_act_y_diff"]
result_df["y_high"] = result_df["avg_act_y_diff"] + result_df["std_act_y_diff"]
df = result_df # shorter
means_df = df[["avg_act_x_diff", "avg_act_y_diff"]]
means_df.columns = ["(PPO - CEsc) X action", "(PPO - CEsc) Y action"]

# dict with errors
x_dict = {
  df.avg_act_x_diff.loc[i]: {'min':df.x_low.loc[i], 'max':df.x_high.loc[i]} for i in df.index
}
y_dict = {
  df.avg_act_y_diff.loc[i]: {'min':df.y_low.loc[i], 'max':df.y_high.loc[i]} for i in df.index
}
results_dict={**x_dict, **y_dict}

# ACTUALLY PLOT

# for X-axis labels
index = list(result_df.bin_range.unique())
means_df.index = index

ax = means_df.plot.bar(rot=0, colormap=newcmp)
ax.yaxis.grid(True, which='major')
plt.yticks([-0.1, 0, 0.1, 0.2, 0.3], minor=False)
plt.xlabel("Time Bin")
plt.ylabel("Difference in Action")
color = 'k'

for p in ax.patches:
  x = p.get_x()  # get the bottom left x corner of the bar
  w = p.get_width()  # get width of bar
  h = p.get_height()  # get height of bar
  min_y = results_dict[h]['min']  # use h to get min from dict z
  max_y = results_dict[h]['max']  # use h to get max from dict z
  plt.vlines(x+w/2, min_y, max_y, color=color)  # draw a vertical line
  plt.hlines(min_y, xmin=x+w/2-0.1, xmax=x+w/2+0.1, color=color)
  plt.hlines(max_y, xmin=x+w/2-0.1, xmax=x+w/2+0.1, color=color)


plt.savefig(fname=f"{DATAPATH}/difference_1.png")





















