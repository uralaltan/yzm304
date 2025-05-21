import gymnasium as gym
from stable_baselines3 import PPO


def visualize(model_path: str = "models/ppo.zip"):
    env = gym.make("CarRacing-v3", render_mode="human")
    model = PPO.load(model_path, env=env)
    obs, _ = env.reset()
    done = False
    while True:
        try:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            env.render()
            if done:
                obs, _ = env.reset()
        except KeyboardInterrupt:
            break
    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/ppo.zip",
                        help="Path to the saved PPO model")
    args = parser.parse_args()
    visualize(args.model)
