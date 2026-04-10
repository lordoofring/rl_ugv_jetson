"""
HSV Tuner — adjust sliders until ONLY the red ball is white in the mask.

Usage:
    python tune_hsv.py --ip <JETSON_IP>       # stream from Jetson
    python tune_hsv.py --camera 0             # local camera

Once tuned, copy the printed values into FrameObserver or config.yaml.
Press S to save current values, Q to quit.
"""

import argparse
import cv2
import numpy as np


def nothing(x):
    pass


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

    cv2.namedWindow("HSV Tuner")
    cv2.namedWindow("Mask")

    # Red wraps around in HSV, so we have two ranges.
    # Range 1: low hue (0-10ish)
    cv2.createTrackbar("H1 Low", "HSV Tuner", 0, 180, nothing)
    cv2.createTrackbar("H1 High", "HSV Tuner", 10, 180, nothing)
    # Range 2: high hue (170-180ish)
    cv2.createTrackbar("H2 Low", "HSV Tuner", 170, 180, nothing)
    cv2.createTrackbar("H2 High", "HSV Tuner", 180, 180, nothing)
    # Shared S and V
    cv2.createTrackbar("S Low", "HSV Tuner", 100, 255, nothing)
    cv2.createTrackbar("S High", "HSV Tuner", 255, 255, nothing)
    cv2.createTrackbar("V Low", "HSV Tuner", 80, 255, nothing)
    cv2.createTrackbar("V High", "HSV Tuner", 255, 255, nothing)

    print("\nAdjust sliders until ONLY the ball is white in the Mask window.")
    print("S = save values, Q = quit\n")

    while True:
        frame = get_frame()
        if frame is None:
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        h1l = cv2.getTrackbarPos("H1 Low", "HSV Tuner")
        h1h = cv2.getTrackbarPos("H1 High", "HSV Tuner")
        h2l = cv2.getTrackbarPos("H2 Low", "HSV Tuner")
        h2h = cv2.getTrackbarPos("H2 High", "HSV Tuner")
        sl = cv2.getTrackbarPos("S Low", "HSV Tuner")
        sh = cv2.getTrackbarPos("S High", "HSV Tuner")
        vl = cv2.getTrackbarPos("V Low", "HSV Tuner")
        vh = cv2.getTrackbarPos("V High", "HSV Tuner")

        mask1 = cv2.inRange(hsv, np.array([h1l, sl, vl]), np.array([h1h, sh, vh]))
        mask2 = cv2.inRange(hsv, np.array([h2l, sl, vl]), np.array([h2h, sh, vh]))
        mask = mask1 | mask2

        mask_clean = cv2.erode(mask, None, iterations=2)
        mask_clean = cv2.dilate(mask_clean, None, iterations=2)

        # Show contour on original frame
        vis = frame.copy()
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            ((cx, cy), radius) = cv2.minEnclosingCircle(largest)
            if radius > 3:
                cv2.circle(vis, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)
                cv2.putText(vis, f"r={radius:.0f}px", (int(cx+radius+5), int(cy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show HSV value at center of frame (helps understand lighting)
        ch, cw = frame.shape[:2]
        center_hsv = hsv[ch//2, cw//2]
        cv2.putText(vis, f"Center HSV: {center_hsv}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow("HSV Tuner", vis)
        cv2.imshow("Mask", mask_clean)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            print(f"\n--- Saved HSV Values ---")
            print(f"ball_hsv_low1=({h1l}, {sl}, {vl}),")
            print(f"ball_hsv_high1=({h1h}, {sh}, {vh}),")
            print(f"ball_hsv_low2=({h2l}, {sl}, {vl}),")
            print(f"ball_hsv_high2=({h2h}, {sh}, {vh}),")
            print(f"------------------------\n")

    cv2.destroyAllWindows()
    if cap:
        cap.release()


if __name__ == "__main__":
    main()
