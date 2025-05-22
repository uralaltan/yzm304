import gymnasium as gym
import argparse
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack, VecNormalize


def make_env(headless):
    mode = "rgb_array" if headless else "human"
    return gym.make("CarRacing-v3", render_mode=mode)


def visualize(algo, model_path, stats_path, headless):
    venv = DummyVecEnv([lambda: make_env(headless)])
    venv = VecTransposeImage(venv)
    venv = VecFrameStack(venv, n_stack=4)
    venv = VecNormalize.load(stats_path, venv)
    venv.training = False
    Model = PPO if algo.lower() == "ppo" else SAC
    model = Model.load(model_path, env=venv)
    obs = venv.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated = venv.step(action)
        venv.render()
        if terminated or truncated:
            obs = venv.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--model", choices=["models/ppo_car_racing.zip", "models/sac_car_racing.zip"],
                        default="models/ppo_car_racing.zip")
    parser.add_argument("--stats", default="models/vecnormalize.pkl")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    visualize(args.algo, args.model, args.stats, args.headless)
