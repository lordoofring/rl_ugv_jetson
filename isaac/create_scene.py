"""
Create the Ball Push scene in Isaac Sim.

Sets up:
  - Ground plane
  - Square arena boundary (visible tape lines)
  - Red ball with physics
  - UGV Rover robot (from USD)
  - Camera on the robot

Run inside the Apptainer container:
    /isaac-sim/python.sh /workspace/rl_ugv_jetson/isaac/create_scene.py
"""

from isaacsim import SimulationApp

app = SimulationApp({"headless": False, "width": 1280, "height": 720})

import omni.kit.commands
import omni.usd
from pxr import Gf, UsdGeom, UsdPhysics, UsdShade, Sdf, PhysxSchema

stage = omni.usd.get_context().get_stage()

ARENA_SIZE = 2.0  # meters
HALF = ARENA_SIZE / 2.0
BALL_RADIUS = 0.05
BALL_START = (0.3, 0.3, BALL_RADIUS)
ROBOT_USD = "/workspace/ugv_rover.usd"

# ------------------------------------------------------------------
# 1. Ground plane
# ------------------------------------------------------------------
omni.kit.commands.execute("CreateMeshPrimWithDefaultXform", prim_type="Plane")
ground = stage.GetPrimAtPath("/World/Plane")
if ground:
    UsdGeom.Xformable(ground).AddScaleOp().Set(Gf.Vec3f(5.0, 5.0, 1.0))
    # Add physics collider to ground
    UsdPhysics.CollisionAPI.Apply(ground)

# ------------------------------------------------------------------
# 2. Physics scene
# ------------------------------------------------------------------
physics_scene_path = "/World/PhysicsScene"
UsdPhysics.Scene.Define(stage, physics_scene_path)
physics_scene = UsdPhysics.Scene.Get(stage, physics_scene_path)
physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
physics_scene.CreateGravityMagnitudeAttr(9.81)

# ------------------------------------------------------------------
# 3. Arena boundary lines (thin boxes as tape markers)
# ------------------------------------------------------------------
TAPE_WIDTH = 0.03
TAPE_HEIGHT = 0.005

arena_root = UsdGeom.Xform.Define(stage, "/World/Arena")

edges = {
    "north": (0.0, HALF, 0.0, ARENA_SIZE / 2, TAPE_WIDTH / 2, TAPE_HEIGHT / 2),
    "south": (0.0, -HALF, 0.0, ARENA_SIZE / 2, TAPE_WIDTH / 2, TAPE_HEIGHT / 2),
    "east": (HALF, 0.0, 0.0, TAPE_WIDTH / 2, ARENA_SIZE / 2, TAPE_HEIGHT / 2),
    "west": (-HALF, 0.0, 0.0, TAPE_WIDTH / 2, ARENA_SIZE / 2, TAPE_HEIGHT / 2),
}

for name, (tx, ty, tz, sx, sy, sz) in edges.items():
    path = f"/World/Arena/{name}"
    cube = UsdGeom.Cube.Define(stage, path)
    xform = UsdGeom.Xformable(cube)
    xform.AddTranslateOp().Set(Gf.Vec3d(tx, ty, tz))
    xform.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))
    # Yellow color for tape
    cube.CreateDisplayColorAttr([(1.0, 0.9, 0.0)])

# ------------------------------------------------------------------
# 4. Red ball with physics
# ------------------------------------------------------------------
ball_path = "/World/Ball"
ball = UsdGeom.Sphere.Define(stage, ball_path)
ball_xform = UsdGeom.Xformable(ball)
ball_xform.AddTranslateOp().Set(Gf.Vec3d(*BALL_START))
ball_xform.AddScaleOp().Set(Gf.Vec3f(BALL_RADIUS, BALL_RADIUS, BALL_RADIUS))
ball.CreateDisplayColorAttr([(0.9, 0.1, 0.1)])

# Ball physics
UsdPhysics.RigidBodyAPI.Apply(ball.GetPrim())
UsdPhysics.CollisionAPI.Apply(ball.GetPrim())
mass_api = UsdPhysics.MassAPI.Apply(ball.GetPrim())
mass_api.CreateMassAttr(0.1)  # 100g ball

# Ball material (friction + bounce)
ball_mat_path = "/World/Ball/PhysicsMaterial"
UsdShade.Material.Define(stage, ball_mat_path)
ball_phys_mat = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(ball_mat_path))
ball_phys_mat.CreateStaticFrictionAttr(0.5)
ball_phys_mat.CreateDynamicFrictionAttr(0.3)
ball_phys_mat.CreateRestitutionAttr(0.4)

# Bind material to ball
binding_api = UsdShade.MaterialBindingAPI.Apply(ball.GetPrim())
binding_api.Bind(
    UsdShade.Material(stage.GetPrimAtPath(ball_mat_path)),
    UsdShade.Tokens.weakerThanDescendants,
    "physics",
)

# ------------------------------------------------------------------
# 5. Import robot
# ------------------------------------------------------------------
robot_path = "/World/Robot"
robot_prim = stage.DefinePrim(robot_path)
robot_prim.GetReferences().AddReference(ROBOT_USD)
robot_xform = UsdGeom.Xformable(robot_prim)
robot_xform.AddTranslateOp().Set(Gf.Vec3d(-0.5, -0.5, 0.08))

# ------------------------------------------------------------------
# 6. Lighting
# ------------------------------------------------------------------
light_path = "/World/DistantLight"
light = UsdGeom.Xform.Define(stage, light_path)
distant_light = stage.DefinePrim(light_path, "DistantLight")
distant_light.CreateAttribute("intensity", Sdf.ValueTypeNames.Float).Set(3000.0)

# ------------------------------------------------------------------
# 7. Save scene
# ------------------------------------------------------------------
scene_path = "/workspace/ball_push_scene.usd"
stage.Export(scene_path)
print(f"\nScene saved to {scene_path}")
print("You can now open this in Isaac Sim: File -> Open -> /workspace/ball_push_scene.usd")

# Keep the app running so you can see the scene
print("\nScene is loaded. Close the window to exit.")
while app.is_running():
    app.update()

app.close()
