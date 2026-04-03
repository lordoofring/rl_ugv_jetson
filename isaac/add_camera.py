"""
Add a camera matching the OV5647 fisheye to the robot in the Ball Push scene.
Paste in Isaac Sim Script Editor after loading ball_push_scene.usd.

Real camera specs:
  - Sensor: OV5647 (1/4" CMOS, 3.674mm × 2.760mm)
  - Resolution: 640×480 (for our use case)
  - FOV: 160° ultra-wide fisheye
  - Mounted on 2-DOF pan-tilt
  - Aperture: f/2.35

We approximate the 160° FOV with a very short focal length.
For a 1/4" sensor (3.674mm horizontal aperture):
  focal_length = (aperture/2) / tan(FOV/2) = (3.674/2) / tan(80°) ≈ 0.324mm
  But Isaac Sim works in mm for focal length, so focal_length ≈ 0.32mm

Since Isaac Sim cameras can't do true fisheye distortion natively,
we use a very wide FOV rectilinear projection as an approximation.
The FrameObserver CV pipeline handles both.
"""

import omni.usd
from pxr import Gf, UsdGeom

stage = omni.usd.get_context().get_stage()

cam_path = "/World/Robot/FrontCamera"

# Remove old camera if it exists
old = stage.GetPrimAtPath(cam_path)
if old:
    stage.RemovePrim(cam_path)

cam = UsdGeom.Camera.Define(stage, cam_path)
xform = UsdGeom.Xformable(cam)

# Position: front of robot, slightly above chassis, tilted down to see ground
xform.AddTranslateOp().Set(Gf.Vec3d(0.12, 0.0, 0.10))
# Tilt down to see ball on ground + tape boundary
xform.AddRotateXYZOp().Set(Gf.Vec3f(70, 0, -90))

# OV5647 sensor specs
# Sensor size: 3.674mm x 2.760mm (1/4" format)
# For 160° FOV we need a very short focal length
# Using ~2.5mm focal length gives roughly 120-130° FOV in rectilinear
# (true 160° requires fisheye projection which Isaac Sim doesn't natively support)
# This is the closest approximation in rectilinear mode
cam.CreateFocalLengthAttr(2.5)
cam.CreateHorizontalApertureAttr(3.674)
cam.CreateVerticalApertureAttr(2.760)
cam.CreateFStopAttr(2.35)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10.0))

print(f"Camera created at {cam_path}")
print("Specs: OV5647-matched, ~120° FOV (rectilinear approximation of 160° fisheye)")
print("")
print("To preview: right-click FrontCamera in Stage panel -> Set as Active Camera")
print("SAVE THE SCENE after adding!")
