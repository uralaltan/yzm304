import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack, VecNormalize


def make_env():
    return gym.make("CarRacing-v3", render_mode="rgb_array")


def build_env(stats_path):
    env = DummyVecEnv([make_env])
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)
    env = VecNormalize.load(stats_path, env)
    env.training = False
    return env


def evaluate(model_class, model_path, stats_path, n_episodes=20):
    env = build_env(stats_path)
    model = model_class.load(model_path, env=env)
    rewards = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated = env.step(action)
            done = terminated or truncated
            total_reward += reward
        rewards.append(total_reward)
    env.close()
    return np.array(rewards)


def main():
    stats_path = "models/vecnormalize.pkl"
    ppo_path = "models/ppo_car_racing.zip"
    sac_path = "models/sac_car_racing.zip"
    td3_path = "models/td3_car_racing.zip"

    ppo_rewards = evaluate(PPO, ppo_path, stats_path)
    sac_rewards = evaluate(SAC, sac_path, stats_path)
    td3_rewards = evaluate(TD3, td3_path, stats_path)

    print(f"PPO    -> mean: {ppo_rewards.mean():.2f}, std: {ppo_rewards.std():.2f}")
    print(f"SAC    -> mean: {sac_rewards.mean():.2f}, std: {sac_rewards.std():.2f}")
    print(f"TD3    -> mean: {td3_rewards.mean():.2f}, std: {td3_rewards.std():.2f}")
    
    print("\nComparisons:")
    sac_ppo_diff = sac_rewards.mean() - ppo_rewards.mean()
    td3_ppo_diff = td3_rewards.mean() - ppo_rewards.mean()
    td3_sac_diff = td3_rewards.mean() - sac_rewards.mean()
    
    print(f"SAC_mean - PPO_mean = {sac_ppo_diff:.2f}")
    print(f"TD3_mean - PPO_mean = {td3_ppo_diff:.2f}")
    print(f"TD3_mean - SAC_mean = {td3_sac_diff:.2f}")
    
    # Find best performing algorithm
    algorithms = ["PPO", "SAC", "TD3"]
    means = [ppo_rewards.mean(), sac_rewards.mean(), td3_rewards.mean()]
    best_idx = np.argmax(means)
    print(f"\nBest performing algorithm: {algorithms[best_idx]} with mean reward: {means[best_idx]:.2f}")


if __name__ == "__main__":
    main()
