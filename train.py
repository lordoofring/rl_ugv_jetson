import argparse
import yaml
import os
import time
import pygame
import numpy as np
import json
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from ugv_rl.envs.grid_map_env import GridMapEnv, ACTION_NAMES
from ugv_rl.controllers.mock_robot import MockRobot
from ugv_rl.controllers.real_robot import RealRobot

def main():
    parser = argparse.ArgumentParser(description='Train or Test RL agent for UGV')
    parser.add_argument('--train', action='store_true', help='Train the agent')
    parser.add_argument('--test', action='store_true', help='Test the agent')
    parser.add_argument('--real', action='store_true', help='Use real robot (Local or Remote)')
    parser.add_argument('--ip', type=str, default=None, help='IP address of Robot Server (if remote)')
    parser.add_argument('--model', type=str, default=None, help='Path to model to load')
    parser.add_argument('--manual', action='store_true', help='Manual WASD control for calibration')
    parser.add_argument('--timesteps', type=int, default=100000, help='Total timesteps for training')
    parser.add_argument('--vision', action='store_true', help='Enable ArUco vision localizer (real robot only)')

    args = parser.parse_args()

    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Create robot instance
    if args.real:
        if args.ip:
            from ugv_rl.controllers.remote_robot import RemoteRobot
            print(f"Connecting to remote robot at {args.ip}...")
            robot = RemoteRobot(ip=args.ip)
        else:
            vision_cfg = config.get('vision', {})
            use_vision = args.vision or vision_cfg.get('enabled', False)
            vision_kwargs = {}
            if use_vision:
                vision_kwargs = {
                    'camera_index': vision_cfg.get('camera_index', 0),
                    'marker_map_path': vision_cfg.get('marker_map_path', 'marker_map.json'),
                    'cell_size': config['env'].get('cell_size', 0.5),
                    'marker_size': vision_cfg.get('marker_size', 0.05),
                    'aruco_dict_name': vision_cfg.get('aruco_dict', 'DICT_4X4_50'),
                }
                print("Using local RealRobot with vision localizer...")
            else:
                print("Using local RealRobot...")
            robot = RealRobot(use_vision=use_vision, vision_kwargs=vision_kwargs)
    else:
        robot = MockRobot()

    # Load Map if available
    map_layout = None
    start_pos = None
    goal_pos = None
    try:
        with open('map.json', 'r') as f:
            map_data = json.load(f)
            map_layout = np.array(map_data['grid'])
            start_pos = np.array(map_data['start']) * config['env']['cell_size']
            goal_pos = np.array(map_data['goal']) * config['env']['cell_size']
            print("Loaded map from map.json")
    except FileNotFoundError:
        print("No map.json found, using empty grid.")

    # Create environment — randomize start/goal during training for generalization
    randomize = args.train
    env = GridMapEnv(config_path='config.yaml', robot=robot, map_layout=map_layout,
                     randomize_positions=randomize)

    if start_pos is not None:
        env.set_map(map_layout, start_pos, goal_pos)

    if args.train:
        print("Starting training...")
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=config['training']['learning_rate'])

        checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/',
                                                 name_prefix='ugv_ppo')

        model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback)
        model.save("ugv_ppo_final")
        print("Training finished.")

    if args.test:
        print("Starting testing...")
        if args.model:
            model = PPO.load(args.model)
        else:
            try:
                model = PPO.load("ugv_ppo_final")
            except:
                print("No model found, creating random agent for testing.")
                model = PPO("MlpPolicy", env)
                model.is_random_agent = True

        # Initialize UI
        from ugv_rl.ui.app import UGVApp
        try:
            app = UGVApp()
            app.mode = 'run'
            app.map_grid = env.map
            app.start_pos = [env.start_gx, env.start_gy]
            app.goal_pos = [env.goal_gx, env.goal_gy]
            print("Visualization initialized.")
        except Exception as e:
            print(f"Visualization failed: {e}")
            app = None

        clock = pygame.time.Clock()
        running = True
        episode_num = 0

        while running:
            obs, _ = env.reset()
            done = False
            truncated = False
            episode_reward = 0.0
            episode_num += 1

            if app:
                app.trail = []
                app.episode_num = episode_num
                app.start_pos = [env.start_gx, env.start_gy]
                app.goal_pos = [env.goal_gx, env.goal_gy]
                app.trail.append((env.agent_gx, env.agent_gy))

            print(f"\n--- Episode {episode_num} ---")
            print(f"Start: ({env.agent_gx}, {env.agent_gy})  Goal: ({env.goal_gx}, {env.goal_gy})")

            while not done and not truncated:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                        running = False
                        print("User closed the window.")
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            done = True
                            running = False

                if not running:
                    break

                use_deterministic = not getattr(model, "is_random_agent", False)
                action, _ = model.predict(obs, deterministic=use_deterministic)
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward

                action_name = ACTION_NAMES.get(int(action), '?')
                print(f"  Step {env.steps}: {action_name} -> ({env.agent_gx},{env.agent_gy})  r={reward:.1f}")

                # Update visualization
                if app:
                    app.agent_cell = (env.agent_gx, env.agent_gy)
                    app.robot_pose = (env.agent_gx * env.cell_size,
                                      env.agent_gy * env.cell_size,
                                      env.agent_theta)
                    app.action_name = action_name
                    app.step_count = env.steps
                    app.episode_reward = episode_reward
                    app.trail.append((env.agent_gx, env.agent_gy))

                    app.render_run_frame()

                # Pace the visualization so you can watch it
                clock.tick(4)  # 4 steps per second for readability

                if done or truncated:
                    result = "GOAL!" if env.agent_gx == env.goal_gx and env.agent_gy == env.goal_gy else "timeout/fail"
                    print(f"  Episode finished: {result}  Total reward: {episode_reward:.1f}")
                    # Pause briefly to see final state
                    time.sleep(1.5)

    if args.manual:
        from ugv_rl.ui.app import UGVApp
        app = UGVApp()
        app.mode = 'run'
        app.map_grid = env.map
        app.start_pos = [env.start_gx, env.start_gy]
        app.goal_pos = [env.goal_gx, env.goal_gy]

        KEY_TO_ACTION = {
            pygame.K_w: 0,  # North
            pygame.K_s: 1,  # South
            pygame.K_d: 2,  # East
            pygame.K_a: 3,  # West
        }

        obs, _ = env.reset()
        episode_reward = 0.0
        app.trail = [(env.agent_gx, env.agent_gy)]
        app.episode_num = 1

        pygame.event.clear()
        print("\n--- Manual Control ---")
        print("CLICK THE PYGAME WINDOW, then use W/A/S/D to move.")
        print("W=North  S=South  A=West  D=East  R=Reset  Q=Quit")
        print(f"Start: ({env.agent_gx}, {env.agent_gy})  Goal: ({env.goal_gx}, {env.goal_gy})")

        running = True
        while running:
            # Update display
            app.agent_cell = (env.agent_gx, env.agent_gy)
            app.robot_pose = (env.agent_gx * env.cell_size,
                              env.agent_gy * env.cell_size,
                              env.agent_theta)
            app.step_count = env.steps
            app.episode_reward = episode_reward
            app.render_run_frame()

            # Wait for a keypress
            action = None
            while action is None and running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            running = False
                        elif event.key == pygame.K_r:
                            obs, _ = env.reset()
                            episode_reward = 0.0
                            app.trail = [(env.agent_gx, env.agent_gy)]
                            app.episode_num += 1
                            app.action_name = 'RESET'
                            print(f"\n--- Reset (Episode {app.episode_num}) ---")
                            break
                        elif event.key in KEY_TO_ACTION:
                            action = KEY_TO_ACTION[event.key]
                pygame.time.wait(30)

            if action is not None:
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                action_name = ACTION_NAMES.get(action, '?')
                app.action_name = action_name
                app.trail.append((env.agent_gx, env.agent_gy))
                print(f"  Step {env.steps}: {action_name} -> ({env.agent_gx},{env.agent_gy})  r={reward:.1f}")

                if done:
                    result = "GOAL!" if env.agent_gx == env.goal_gx and env.agent_gy == env.goal_gy else "done"
                    print(f"  Episode finished: {result}  Total reward: {episode_reward:.1f}")
                    print("  Press R to reset or Q to quit.")

    if args.real:
        robot.close()

if __name__ == '__main__':
    main()
