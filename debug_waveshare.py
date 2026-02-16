import serial
import json
import queue
import threading
import time
import os

# Exact copy of ReadLine from vendor
class ReadLine:
    def __init__(self, s):
        self.buf = bytearray()
        self.s = s

    def readline(self):
        i = self.buf.find(b"\n")
        if i >= 0:
            r = self.buf[:i+1]
            self.buf = self.buf[i+1:]
            return r
        while True:
            i = max(1, min(512, self.s.in_waiting))
            data = self.s.read(i)
            i = data.find(b"\n")
            if i >= 0:
                r = self.buf + data[:i+1]
                self.buf[0:] = data[i+1:]
                return r
            else:
                self.buf.extend(data)

# Modified version of BaseController to be standalone (removed config.yaml dependency for simplicity)
class BaseController:
    def __init__(self, uart_dev_set, buad_set):
        self.ser = serial.Serial(uart_dev_set, buad_set, timeout=1)
        self.rl = ReadLine(self.ser)
        self.command_queue = queue.Queue()
        self.command_thread = threading.Thread(target=self.process_commands, daemon=True)
        self.command_thread.start()

    def on_data_received(self):
        try:
            line = self.rl.readline()
            if line:
                data_read = json.loads(line.decode('utf-8'))
                self.ser.reset_input_buffer()
                return data_read
        except:
            pass
        return None

    def send_command(self, data):
        self.command_queue.put(data)

    def process_commands(self):
        while True:
            data = self.command_queue.get()
            # Encoded exactly as vendor
            msg = json.dumps(data) + '\n'
            # print(f"DEBUG SEND: {msg.strip()}")
            self.ser.write(msg.encode("utf-8"))

    def base_speed_ctrl(self, input_left, input_right):
        # T:1 is the command for speed control
        data = {"T":1,"L":int(input_left),"R":int(input_right)}
        self.send_command(data)

    def close(self):
        self.ser.close()

if __name__ == '__main__':
    # Use the port you confirmed
    port = '/dev/ttyTHS1' 
    baud = 115200
    
    print(f"Opening serial port {port} at {baud}...")
    try:
        ctrl = BaseController(port, baud)
    except Exception as e:
        print(f"FAILED to open serial port: {e}")
        exit(1)

    print("Sending Forward command (L:-100, R:100) - Assuming differential drive directions...")
    # Note: Depending on wiring, one might need to be inverted.
    # Vendor code input_left, input_right.
    
    try:
        # Loop to keep sending commands
        for i in range(10):
            print(f"Step {i}: Sending speed 100...")
            ctrl.base_speed_ctrl(100, 100) # Try both positive first
            
            # Read any feedback
            # feedback = ctrl.on_data_received()
            # if feedback:
            #     print(f"Feedback: {feedback}")
                
            time.sleep(0.5)
            
        print("Stopping...")
        ctrl.base_speed_ctrl(0, 0)
        time.sleep(1)
        
    except KeyboardInterrupt:
        pass
    finally:
        ctrl.close()
