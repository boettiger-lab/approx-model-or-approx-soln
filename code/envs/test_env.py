env = fish_tipping(config = {"Tmax": 200})
env.Tmax
#env.threshold = 0.00
env.training = False
env.reset()
env.state = np.array([-0.5,-1,-1], dtype=np.float32)

env.state = np.array([-1,-0.5,-1], dtype=np.float32)

env.state = np.array([-0.7,-1,-0.5], dtype=np.float32)

env.population()

df = []
episode_reward = 0
action = 0.01
observation, _ = env.reset()
for t in range(env.Tmax):
  df.append(np.append([t, action, episode_reward], observation))
  observation, reward, terminated, done, info = env.step(action)
  episode_reward += reward
  if terminated:
    break
  
import pandas as pd
cols = ["t", "action", "reward", "X", "Y", "Z"]
df = pd.DataFrame(df, columns = cols)
env.population()
