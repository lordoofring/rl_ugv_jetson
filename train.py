import argparse
import yaml
import os
import time

import numpy as np
import json
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from ugv_rl.envs.grid_map_env import GridMapEnv
from ugv_rl.controllers.mock_robot import MockRobot
from ugv_rl.controllers.real_robot import RealRobot
from ugv_rl.ui.app import BLACK, WHITE, GRAY, RED, GREEN, BLUE

def main():
    parser = argparse.ArgumentParser(description='Train or Test RL agent for UGV')
    parser.add_argument('--train', action='store_true', help='Train the agent')
    parser.add_argument('--test', action='store_true', help='Test the agent')
    parser.add_argument('--real', action='store_true', help='Use real robot (Local or Remote)')
    parser.add_argument('--ip', type=str, default=None, help='IP address of Robot Server (if remote)')
    parser.add_argument('--model', type=str, default=None, help='Path to model to load')
    parser.add_argument('--timesteps', type=int, default=100000, help='Total timesteps for training')
    
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
            print("Using local RealRobot...")
            robot = RealRobot()
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

    # Create environment
    # Note: make_vec_env automatically wraps the env. 
    # For custom env with init args, needed a callable.
    env = GridMapEnv(config_path='config.yaml', robot=robot, map_layout=map_layout)
    
    if start_pos is not None:
        env.set_map(map_layout, start_pos, goal_pos)
    
    if args.train:
        print("Starting training...")
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=config['training']['learning_rate'])
        
        # Save checkpoints
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
            # If no model provided, use a fresh one (bad performance) or load default
            try:
                model = PPO.load("ugv_ppo_final")
            except:
                print("No model found, creating random agent for testing.")
                model = PPO("MlpPolicy", env) # untrained
                
        obs, _ = env.reset()
        done = False
        
        # for visualization
        import pygame
        from ugv_rl.ui.app import UGVApp
        app = UGVApp()
        app.mode = 'run'
        app.map_grid = env.map
        # Convert absolute metric positions back to grid indices for UI
        # This is a bit hacky, UI should support metric or we sync better
        # For now, just visualizing map. Robot update needed.

        while not done:
            # Handle Pygame events to allow closing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            # Update UI
            state = env.robot.get_state()
            app.robot_pose = (state['x'], state['y'], state['theta'])
            
            app.screen.fill(GRAY)
            app.draw_grid()
            app.draw_robot()
            pygame.display.flip()
            # clock tick?
            
            if done or truncated:
                print(f"Episode finished. Info: {info}")
                obs, _ = env.reset()
                break # Just run one episode for test
                
    if args.real:
        robot.close()

if __name__ == '__main__':
    main()
