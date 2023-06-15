import pandas as pd
import seaborn as sns
from plotnine import ggplot, aes, geom_bar, geom_point, geom_line
import os

from msy_fns import csv_to_frac_msy_1fish, csv_to_frac_msy_2fish

#globals
1sp1fish = ["data", "results_data", "ONESP", "high_beta", "msy_100.csv.xz"]
3sp1fish = ["data", "results_data", "1FISHERY", "DEFAULT", "msy_100.csv.xz"]
3sp2fish = ["data", "results_data", "2FISHERY", "DEFAULT", "msy_100.csv.xz"]
3sp2fish_timevar = ["data", "results_data", "2FISHERY", "RXDRIFT", "msy_100.csv.xz"]

fname = os.path.join(
  *1sp1fish, 
)
