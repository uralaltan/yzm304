import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecTransposeImage,
    VecFrameStack,
    VecNormalize,
)

def make_env():
    return gym.make("CarRacing-v3", render_mode="human")

def main():
    # Vectorize and prepare frames for PyTorch
    venv = DummyVecEnv([make_env])
    venv = VecTransposeImage(venv)  # HxWxC → CxHxW
    venv = VecFrameStack(venv, n_stack=4)
    # Normalize obs but do NOT normalize images
    venv = VecNormalize(
        venv,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )

    # Create PPO with CNN policy, disabling image normalization again
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
        policy_kwargs={"normalize_images": False},  # policy-level flag
    )

    # Train for 1M timesteps
    model.learn(total_timesteps=1_000_000)

    # Evaluate and render 3 episodes
    for ep in range(3):
        obs, _ = venv.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = venv.step(action)
            done = terminated or truncated
            total_reward += reward
            venv.render()
        print(f"Episode {ep+1} reward: {total_reward}")

    venv.close()

if __name__ == "__main__":
    main()
