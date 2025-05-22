import os
import gymnasium as gym
from stable_baselines3 import SAC
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

    venv = DummyVecEnv([lambda: make_env(headless)])
    venv = VecTransposeImage(venv)
    venv = VecFrameStack(venv, n_stack=4)
    venv = VecNormalize(
        venv,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )

    model = SAC(
        "CnnPolicy",
        venv,
        verbose=1,
        buffer_size=10_000,
        optimize_memory_usage=True,
        replay_buffer_kwargs={
            "handle_timeout_termination": False
        },
        train_freq=(1, "step"),
        gradient_steps=1,
        learning_rate=3e-4,
        batch_size=256,
        tau=0.005,
        ent_coef="auto",
        gamma=0.99,
        policy_kwargs={"normalize_images": False},
    )

    model.learn(total_timesteps=1_000)

    os.makedirs("models", exist_ok=True)
    model.save("models/sac_car_racing")

    if not headless:
        for ep in range(3):
            obs = venv.reset()
            done = False
            total_reward = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated = venv.step(action)
                done = terminated or truncated
                total_reward += reward
                venv.render()
            print(f"Episode {ep + 1} reward: {total_reward}")

    venv.close()


if __name__ == "__main__":
    main()
