
## plot locally for fun
import pandas as pd
from plotnine import ggplot, geom_point, aes, geom_line, facet_wrap, geom_path

df = pd.read_csv("data/PPO.csv.gz")

df2 = (df
       .melt(id_vars=["t", "action", "reward", "rep"])
       .groupby(['t', "variable"], as_index=False)
       .agg({'reward': 'mean',
             'value': 'mean',
             'action': 'mean'})) 


(ggplot(df2, aes("t", "value", color="variable")) +
 geom_line())
(ggplot(df2, aes("t", "action", color="variable")) + geom_line())
(ggplot(df2, aes("t", "reward", color="variable")) + geom_line())

(ggplot(df, aes("sp2", "sp3", color="action", size="sp1")) + geom_point())
(ggplot(df, aes("sp1", "sp2", color="action", size="sp3")) + geom_point())
(ggplot(df, aes("sp1", "sp3", color="action", size="sp2")) + geom_point())

