from parameters import parameters
from envs import two_three_fishing
from train_fns import create_agent, train_agent
from uncontrolled_fns import generate_uncontrolled_timeseries_plot
#from escapement_fns import blah
from util import dict_pretty_print, print_params
from eval_util import generate_episodes, episode_plots, state_policy_plot

import os

'''
Data-code list:

"DEFAULT","RXDRIFT","V0DRIFT","DDRIFT","KLIMIT","KLIMIT_RXDRIFT","BETADRIFT","CVDRIFT","YABIOTIC","ZABIOTIC","COUPLFLUC",
'''

# globals
ITERATIONS = 100
REPS = 100
DATACODE = "RXDRIFT"
ENVCODE = "2FISHERY"
DATAPATH = os.path.join("../data/results_data", ENVCODE, DATACODE)

os.makedirs(DATAPATH, exist_ok=True)

# define problem
env_class = two_three_fishing.twoThreeFishing
parameters_obj = parameters()

# create agent
agent = create_agent(
  env_class = env_class,
  parameters_obj=parameters_obj,
  datacode = "RXDRIFT", 
  env_name="twoThreeFishing", 
  fluctuating=True, 
  training=True,
)

# create associated env
env_config = agent.evaluation_config.env_config
env_config.update({'seed': 42})
env = agent.env_creator(env_config)

# uncontrolled
generate_uncontrolled_timeseries_plot(env, path_and_filename=f"{DATAPATH}/uncontrolled_timeseries.png", T=200, reps=1)

# escapement

# train agent and generate data
agent = train_agent(agent, iterations=ITERATIONS, path_to_checkpoint="cache")
ppo_df = generate_episodes(agent, env, reps = REPS)
ppo_df.to_csv(f"{DATAPATH}/ppo{ITERATIONS}.csv.xz", index = False)

# plots

