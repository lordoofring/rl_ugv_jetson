"""
Create the Ball Push scene in Isaac Sim.

Run inside the Apptainer container:
    /isaac-sim/python.sh /workspace/rl_ugv_jetson/isaac/create_scene.py
"""

from isaacsim import SimulationApp

app = SimulationApp({"headless": False, "width": 1280, "height": 720})

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicSphere, FixedCuboid, GroundPlane
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.stage import add_reference_to_stage
import numpy as np
import omni.usd

ARENA_SIZE = 2.0
HALF = ARENA_SIZE / 2.0
BALL_RADIUS = 0.05
ROBOT_USD = "/workspace/ugv_rover.usd"

# ------------------------------------------------------------------
# Create world with ground plane and lighting (handled automatically)
# ------------------------------------------------------------------
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

# ------------------------------------------------------------------
# Arena boundary (4 yellow tape lines as thin cuboids)
# ------------------------------------------------------------------
TAPE_W = 0.03
TAPE_H = 0.005

world.scene.add(FixedCuboid(
    prim_path="/World/Arena/north",
    name="tape_north",
    size=1.0,
    scale=np.array([ARENA_SIZE, TAPE_W, TAPE_H]),
    position=np.array([0.0, HALF, TAPE_H / 2]),
    color=np.array([1.0, 0.9, 0.0]),
))
world.scene.add(FixedCuboid(
    prim_path="/World/Arena/south",
    name="tape_south",
    size=1.0,
    scale=np.array([ARENA_SIZE, TAPE_W, TAPE_H]),
    position=np.array([0.0, -HALF, TAPE_H / 2]),
    color=np.array([1.0, 0.9, 0.0]),
))
world.scene.add(FixedCuboid(
    prim_path="/World/Arena/east",
    name="tape_east",
    size=1.0,
    scale=np.array([TAPE_W, ARENA_SIZE, TAPE_H]),
    position=np.array([HALF, 0.0, TAPE_H / 2]),
    color=np.array([1.0, 0.9, 0.0]),
))
world.scene.add(FixedCuboid(
    prim_path="/World/Arena/west",
    name="tape_west",
    size=1.0,
    scale=np.array([TAPE_W, ARENA_SIZE, TAPE_H]),
    position=np.array([-HALF, 0.0, TAPE_H / 2]),
    color=np.array([1.0, 0.9, 0.0]),
))

# ------------------------------------------------------------------
# Red ball with physics
# ------------------------------------------------------------------
world.scene.add(DynamicSphere(
    prim_path="/World/Ball",
    name="ball",
    radius=BALL_RADIUS,
    mass=0.1,
    position=np.array([0.3, 0.3, BALL_RADIUS + 0.01]),
    color=np.array([0.9, 0.1, 0.1]),
))

# ------------------------------------------------------------------
# Robot (from saved USD)
# ------------------------------------------------------------------
add_reference_to_stage(usd_path=ROBOT_USD, prim_path="/World/Robot")
world.scene.add(XFormPrim(
    prim_path="/World/Robot",
    name="robot",
    position=np.array([-0.5, -0.5, 0.08]),
))

# ------------------------------------------------------------------
# Initialize and save
# ------------------------------------------------------------------
world.reset()

# Save the scene
stage = omni.usd.get_context().get_stage()
scene_path = "/workspace/ball_push_scene.usd"
stage.Export(scene_path)
print(f"\nScene saved to {scene_path}")

# Keep running so you can look around
print("Scene loaded. Close the window or Ctrl+C to exit.")
while app.is_running():
    world.step(render=True)

app.close()
