import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecTransposeImage,
    VecFrameStack,
    VecNormalize,
)


def make_env(headless):
    mode = "rgb_array" if headless else "human"
    return gym.make("CarRacing-v3", render_mode=mode)


def main():
    headless = True
    stats_path = "models/vecnormalize.pkl"

    base_env = DummyVecEnv([lambda: make_env(headless)])
    env = VecTransposeImage(base_env)
    env = VecFrameStack(env, n_stack=4)
    if os.path.exists(stats_path):
        env = VecNormalize.load(stats_path, env)
        env.training = not headless
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        n_steps=256,
        batch_size=64,
        n_epochs=4,
        learning_rate=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        policy_kwargs={"normalize_images": False},
    )

    model.learn(total_timesteps=1_000)
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_car_racing")
    env.save(stats_path)

    if not headless:
        for ep in range(3):
            obs = env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated = env.step(action)
                done = terminated or truncated
                total_reward += reward
                env.render()
            print(f"Episode {ep + 1} reward: {total_reward}")

    env.close()


if __name__ == "__main__":
    main()
