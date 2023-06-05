from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes

class MyCallback(tune.Callback):
    def on_train_result(self, *args, **kwargs):
        result = kwargs["result"]
        episodes, _ = collect_episodes(
            self.trainer,
            self.trainer.workers.local_worker(),
            num_episodes=result["episodes_this_iter"])
        summary = summarize_episodes(episodes)
        print(f"Mean episode reward: {summary['episode_reward_mean']}")
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        # Access the episode rewards and print them out
        rewards = episode["rewards"]
        mean_reward = sum(rewards) / len(rewards)
        print("Mean episode reward:", mean_reward)  
