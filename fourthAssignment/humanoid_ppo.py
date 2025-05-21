import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def main():
    # 1) Create and normalize the Humanoid-v5 environment
    venv = DummyVecEnv([lambda: gym.make("Humanoid-v5", render_mode="human")])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # 2) Instantiate PPO with tuned hyperparameters
    model = PPO(
        policy="MlpPolicy",
        env=venv,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=10,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.0,
    )

    # 3) Train for 5 million timesteps
    model.learn(total_timesteps=5_000_000)

    # 4) Evaluate and render 5 episodes
    for episode in range(5):
        obs, info = venv.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = venv.step(action)
            done = terminated or truncated
            total_reward += reward
            venv.render()
        print(f"Episode {episode + 1} reward: {total_reward}")

    venv.close()


if __name__ == "__main__":
    main()
