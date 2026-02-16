from ugv_rl.envs.grid_nav_env import GridNavEnv
from ugv_rl.controllers.mock_robot import MockRobot
import time

def test_env():
    print("Initializing environment with MockRobot...")
    robot = MockRobot()
    env = GridNavEnv(config_path='config.yaml', robot=robot)
    
    print("Resetting environment...")
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    
    print("Running 10 steps...")
    for i in range(10):
        # Random action
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Done={done}")
        print(f"Robot State: {robot.get_state()}")
        
        if done:
            print("Episode finished.")
            break
            
    print("Environment test passed.")

if __name__ == "__main__":
    test_env()
