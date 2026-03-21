"""Train or test PPO agent for the Ball Push task."""

import argparse
import yaml
import time
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from ugv_rl.envs.ball_push_env import BallPushEnv, ACTION_NAMES
from ugv_rl.controllers.mock_robot import MockRobot


def main():
    parser = argparse.ArgumentParser(description="Ball Push RL Agent")
    parser.add_argument("--train", action="store_true", help="Train the agent")
    parser.add_argument("--test", action="store_true", help="Test the agent")
    parser.add_argument("--model", type=str, default=None, help="Model path to load")
    parser.add_argument("--timesteps", type=int, default=300000, help="Training timesteps")
    parser.add_argument("--manual", action="store_true", help="Manual control")
    args = parser.parse_args()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    robot = MockRobot()
    env = BallPushEnv(config_path="config.yaml", robot=robot)
    arena_size = config.get("ball_push", {}).get("arena_size", 2.0)

    if args.train:
        print("Training Ball Push agent...")
        model = PPO(
            "MlpPolicy", env, verbose=1,
            learning_rate=config["training"]["learning_rate"],
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
        )
        cb = CheckpointCallback(save_freq=25000, save_path="./models/", name_prefix="ball_push_ppo")
        model.learn(total_timesteps=args.timesteps, callback=cb)
        model.save("ball_push_ppo_final")
        print("Training finished. Model saved to ball_push_ppo_final.zip")

    if args.test or args.manual:
        from ugv_rl.ui.ball_push_app import BallPushApp
        app = BallPushApp(arena_size=arena_size)

        model = None
        if args.test:
            if args.model:
                model = PPO.load(args.model)
            else:
                try:
                    model = PPO.load("ball_push_ppo_final")
                except Exception:
                    print("No model found, using random actions.")
                    model = PPO("MlpPolicy", env)

        KEY_TO_ACTION = {
            pygame.K_a: 0,  # Rotate left
            pygame.K_d: 1,  # Rotate right
            pygame.K_w: 2,  # Forward
        }

        clock = pygame.time.Clock()
        running = True
        episode_num = 0

        while running:
            obs, _ = env.reset()
            done = False
            episode_reward = 0.0
            episode_num += 1
            app.episode_num = episode_num
            app.trail = []

            print(f"\n--- Episode {episode_num} ---")
            print(f"Robot: ({env.robot_x:.2f}, {env.robot_y:.2f})  Ball: ({env.ball_x:.2f}, {env.ball_y:.2f})")

            while not done and running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        done = True  # force reset

                if not running:
                    break

                action = None
                if args.manual:
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_a]:
                        action = 0
                    elif keys[pygame.K_d]:
                        action = 1
                    elif keys[pygame.K_w]:
                        action = 2
                else:
                    action, _ = model.predict(obs, deterministic=True)
                    action = int(action)

                if action is not None:
                    obs, reward, done, _, info = env.step(action)
                    episode_reward += reward

                    app.robot_pos = (env.robot_x, env.robot_y)
                    app.robot_theta = env.robot_theta
                    app.ball_pos = (env.ball_x, env.ball_y)
                    app.step_count = env.steps
                    app.episode_reward = episode_reward
                    app.action_name = ACTION_NAMES.get(action, "?")
                    app.trail.append((env.robot_x, env.robot_y))

                app.render()

                if args.manual:
                    clock.tick(15)
                else:
                    clock.tick(30)

                if done:
                    result = "BALL OUT!" if info.get("ball_outside") else "timeout"
                    print(f"  {result}  Reward: {episode_reward:.1f}  Steps: {env.steps}")
                    time.sleep(1.0)

        pygame.quit()


if __name__ == "__main__":
    main()
