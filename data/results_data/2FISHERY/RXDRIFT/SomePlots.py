import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
#mpl.colormaps['viridis']
#colormap = 'gist_rainbow'

def plot_df(df, fname):
  plt.close()
  df.plot()
  plt.savefig(fname)
  plt.close()

esc = pd.read_csv("esc_100.csv.xz")
msy = pd.read_csv("msy_100.csv.xz")

esc_lst = [esc[esc.rep == i].set_index('t')[["X","Y","Z"]] for i in range(3)]
msy_lst = [msy[msy.rep == i].set_index('t')[["X","Y","Z"]] for i in range(3)]


for i in range(3): 
  plot_df(esc_lst[i], f"esc_{i}.png")
  plot_df(msy_lst[i], f"msy_{i}.png")
