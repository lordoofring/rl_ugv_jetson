import serial
import json
import time

def main():
    port = '/dev/ttyTHS1'
    baud = 115200
    
    print(f"Opening {port} at {baud}...")
    try:
        ser = serial.Serial(port, baud, timeout=1)
    except Exception as e:
        print(f"Error opening port: {e}")
        return

    # Helper to send
    def send_json(data):
        msg = json.dumps(data) + '\n'
        print(f"Sending: {msg.strip()}")
        ser.write(msg.encode('utf-8'))
        
    try:
        # 1. Try Lights (Known to work in vendor demo)
        print("Testing Lights (IO4=255, IO5=0)...")
        send_json({"T":132,"IO4":255,"IO5":0})
        time.sleep(1.0)
        
        print("Testing Lights (IO4=0, IO5=255)...")
        send_json({"T":132,"IO4":0,"IO5":255})
        time.sleep(1.0)
        
        # 2. Try Motors
        print("Testing Motors (Speed 150)...")
        # Try sending continuously for a bit, as some robots timeout safety
        for _ in range(5):
            send_json({"T":1,"L":0.2,"R":0.2})
            
            # Check for read
            if ser.in_waiting:
                try:
                    line = ser.readline()
                    print(f"Received: {line.decode('utf-8', errors='ignore').strip()}")
                except:
                    pass
            time.sleep(0.2)
            
        print("Stopping Motors...")
        send_json({"T":1,"L":0,"R":0})
        
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        ser.close()
        print("Closed.")

if __name__ == '__main__':
    main()
