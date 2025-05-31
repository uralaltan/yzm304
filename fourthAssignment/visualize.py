import gymnasium as gym
import argparse
import os
from stable_baselines3 import PPO, SAC, TD3
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
    if algo.lower() == "ppo":
        Model = PPO
    elif algo.lower() == "sac":
        Model = SAC
    elif algo.lower() == "td3":
        Model = TD3
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")
    model = Model.load(model_path, env=venv)
    obs = venv.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated = venv.step(action)
        venv.render()
        if terminated or truncated:
            obs = venv.reset()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "sac", "td3"], default="ppo")
    parser.add_argument("--model", choices=["models/ppo_car_racing.zip", "models/sac_car_racing.zip", "models/td3_car_racing.zip"],
                        default="models/ppo_car_racing.zip")
    parser.add_argument("--stats", default="models/vecnormalize.pkl")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    
    model_path = os.path.join(script_dir, args.model) if not os.path.isabs(args.model) else args.model
    stats_path = os.path.join(script_dir, args.stats) if not os.path.isabs(args.stats) else args.stats
    
    visualize(args.algo, model_path, stats_path, args.headless)
