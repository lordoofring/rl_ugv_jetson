"""
HSV Tuner — adjust sliders to isolate the yellow ball.

Usage:
    python tune_hsv.py --ip <JETSON_IP>
    python tune_hsv.py --camera 0

Controls: Q = quit, S = save values to terminal
"""

import argparse
import cv2
import numpy as np


def nothing(x):
    pass


SLIDER_NAMES = [
    ("H1 Low",  20),
    ("H1 High", 35),
    ("H2 Low",  20),
    ("H2 High", 35),
    ("S Low",   100),
    ("S High",  255),
    ("V Low",   100),
    ("V High",  255),
]

PANEL_W = 300
PANEL_H = 480


def draw_panel(sliders):
    """Draw a dark panel with slider names and current values as readable text."""
    panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)

    cv2.putText(panel, "HSV TUNER", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.putText(panel, "Adjust until ball = white in mask", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    y = 90
    colors = {
        "H1": (0, 180, 255),   # orange - hue range 1
        "H2": (0, 130, 255),   # dark orange - hue range 2
        "S":  (0, 255, 180),   # green - saturation
        "V":  (255, 200, 100), # blue - value/brightness
    }

    for name, val in sliders:
        prefix = name.split(" ")[0]
        color = colors.get(prefix, (255, 255, 255))

        cv2.putText(panel, f"{name}:", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        cv2.putText(panel, f"{val}", (200, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Draw a small bar showing the value
        max_val = 180 if name.startswith("H") else 255
        bar_w = int((val / max_val) * 250)
        cv2.rectangle(panel, (10, y + 5), (10 + bar_w, y + 15), color, -1)
        cv2.rectangle(panel, (10, y + 5), (260, y + 15), (80, 80, 80), 1)

        y += 45

    cv2.putText(panel, "S=save  Q=quit", (10, PANEL_H - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    return panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default=None)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--camera", type=int, default=None)
    args = parser.parse_args()

    cap = None
    get_frame = None

    if args.ip:
        from ugv_rl.controllers.remote_robot import RemoteRobot
        robot = RemoteRobot(ip=args.ip, port=args.port)
        get_frame = robot.get_frame
        print(f"Streaming from {args.ip}")
    elif args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        def cam_frame():
            ret, f = cap.read()
            return f if ret else None
        get_frame = cam_frame
        print(f"Using local camera {args.camera}")
    else:
        print("Specify --ip <JETSON_IP> or --camera <INDEX>")
        return

    # Create slider window (wide enough to show labels)
    win = "Sliders"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 600, 350)

    for name, default in SLIDER_NAMES:
        max_val = 180 if name.startswith("H") else 255
        cv2.createTrackbar(name, win, default, max_val, nothing)

    print("\n=== HSV Tuner ===")
    print("Adjust sliders until ONLY the ball is white in the mask.")
    print("S = save values, Q = quit\n")

    while True:
        frame = get_frame()
        if frame is None:
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Read slider values
        vals = {}
        for name, _ in SLIDER_NAMES:
            vals[name] = cv2.getTrackbarPos(name, win)

        # Build masks
        mask1 = cv2.inRange(
            hsv,
            np.array([vals["H1 Low"],  vals["S Low"],  vals["V Low"]]),
            np.array([vals["H1 High"], vals["S High"], vals["V High"]]),
        )
        mask2 = cv2.inRange(
            hsv,
            np.array([vals["H2 Low"],  vals["S Low"],  vals["V Low"]]),
            np.array([vals["H2 High"], vals["S High"], vals["V High"]]),
        )
        mask = mask1 | mask2
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # Draw detection on camera feed
        vis = frame.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            ((cx, cy), radius) = cv2.minEnclosingCircle(largest)
            if radius > 3:
                cv2.circle(vis, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)
                cv2.putText(vis, "BALL", (int(cx - 20), int(cy - radius - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Build the info panel with labeled values
        slider_vals = [(name, vals[name]) for name, _ in SLIDER_NAMES]
        panel = draw_panel(slider_vals)

        # Resize panel height to match camera frame
        h_frame = vis.shape[0]
        panel_resized = cv2.resize(panel, (PANEL_W, h_frame))

        # Combine: camera feed + panel side by side
        combined = np.hstack([vis, panel_resized])

        # Mask view (resize to match)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        cv2.imshow("Camera + HSV Info", combined)
        cv2.imshow("Mask (ball = white)", mask_bgr)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            h1l, h1h = vals["H1 Low"], vals["H1 High"]
            h2l, h2h = vals["H2 Low"], vals["H2 High"]
            sl, sh = vals["S Low"], vals["S High"]
            vl, vh = vals["V Low"], vals["V High"]
            print(f"\n{'='*45}")
            print(f"  COPY INTO FrameObserver __init__:")
            print(f"{'='*45}")
            print(f"  ball_hsv_low1=({h1l}, {sl}, {vl}),")
            print(f"  ball_hsv_high1=({h1h}, {sh}, {vh}),")
            print(f"  ball_hsv_low2=({h2l}, {sl}, {vl}),")
            print(f"  ball_hsv_high2=({h2h}, {sh}, {vh}),")
            print(f"{'='*45}\n")

    cv2.destroyAllWindows()
    if cap:
        cap.release()


if __name__ == "__main__":
    main()
