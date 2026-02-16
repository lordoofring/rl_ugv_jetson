import time
import sys
import os
import serial

# Ensure the current directory is in python path to find ugv_rl
sys.path.append(os.getcwd())

from ugv_rl.controllers.real_robot import RealRobot

def main():
    port = '/dev/ttyTHS1'
    if len(sys.argv) > 1:
        port = sys.argv[1]

    print(f"Initializing RealRobot on {port}...")
    try:
        robot = RealRobot(serial_port=port)
    except Exception as e:
        print(f"Error initializing robot on {port}: {e}")
        print("Trying fallback ports...")
        fallback_ports = ['/dev/ttyACM0', '/dev/ttyUSB0', '/dev/serial0']
        for p in fallback_ports:
            if p == port: continue
            print(f"Trying {p}...")
            try:
                robot = RealRobot(serial_port=p)
                port = p
                print(f"Success on {p}!")
                break
            except:
                pass
        else:
            print("Could not connect to any common serial port.")
            return

    print("WARNING: Robot will move! Ensure it is safe.")
    print("Press ENTER to start the test (or Ctrl+C to cancel)...")
    try:
        input()
    except KeyboardInterrupt:
        return

    try:
        print("Moving FORWARD at 0.2 m/s...")
        robot.move(0.2, 0.0) # 0.2 m/s linear, 0 angular
        time.sleep(2.0)
        
        print("STOPPING...")
        robot.stop()
        time.sleep(1.0)
        
        print("Turning LEFT at 0.5 rad/s...")
        robot.move(0.0, 0.5) 
        time.sleep(2.0)
        
        print("STOPPING...")
        robot.stop()
        time.sleep(1.0)
        
        print("Test Complete.")
        
    except KeyboardInterrupt:
        print("\nTest cancelled by user. Stopping robot...")
        robot.stop()
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        robot.stop()
    finally:
        robot.close()
        print("Connection closed.")

if __name__ == '__main__':
    main()
