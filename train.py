import argparse
import yaml
import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from ugv_rl.envs.grid_nav_env import GridNavEnv
from ugv_rl.controllers.mock_robot import MockRobot
from ugv_rl.controllers.real_robot import RealRobot

def main():
    parser = argparse.ArgumentParser(description='Train or Test RL agent for UGV')
    parser.add_argument('--train', action='store_true', help='Train the agent')
    parser.add_argument('--test', action='store_true', help='Test the agent')
    parser.add_argument('--real', action='store_true', help='Use real robot')
    parser.add_argument('--model', type=str, default=None, help='Path to model to load')
    parser.add_argument('--timesteps', type=int, default=100000, help='Total timesteps for training')
    
    args = parser.parse_args()
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    # Create robot instance
    if args.real:
        robot = RealRobot()
    else:
        robot = MockRobot()
        
    # Create environment
    # Note: make_vec_env automatically wraps the env. 
    # For custom env with init args, needed a callable.
    env = GridNavEnv(config_path='config.yaml', robot=robot)
    
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
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
            if done or truncated:
                obs, _ = env.reset()
                break # Just run one episode for test
                
    if args.real:
        robot.close()

if __name__ == '__main__':
    main()
