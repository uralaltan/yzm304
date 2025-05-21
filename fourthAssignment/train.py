import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecTransposeImage,
    VecFrameStack,
    VecNormalize,
)


def train(total_timesteps: int, show_animation: bool = False):
    mode = "human" if show_animation else "rgb_array"

    def make_env():
        return gym.make("CarRacing-v3", render_mode=mode)

    venv = DummyVecEnv([make_env])
    venv = VecTransposeImage(venv)
    venv = VecFrameStack(venv, n_stack=4)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    model = PPO(
        "CnnPolicy",
        venv,
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
    model.learn(total_timesteps=total_timesteps)
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo.zip")
    venv.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="Total training timesteps")
    parser.add_argument("--show_animation", action="store_true",
                        help="Render training in a window")
    args = parser.parse_args()
    train(args.timesteps, args.show_animation)
