import serial
import json
import queue
import threading
import time

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


class WaveshareController:
    def __init__(self, uart_dev_set, buad_set):
        self.ser = serial.Serial(uart_dev_set, buad_set, timeout=1)
        self.rl = ReadLine(self.ser)
        self.command_queue = queue.Queue()
        self.command_thread = threading.Thread(target=self.process_commands, daemon=True)
        self.command_thread.start()

    def on_data_received(self):
        try:
            # Non-blocking read attempt not trivial with ReadLine blocking
            # But the Thread is writing. We are reading in main thread.
            if self.ser.in_waiting:
                line = self.rl.readline()
                if line:
                    data_read = json.loads(line.decode('utf-8'))
                    return data_read
        except Exception as e:
            pass
        return None

    def send_command(self, data):
        self.command_queue.put(data)

    def process_commands(self):
        while True:
            data = self.command_queue.get()
            try:
                msg = json.dumps(data) + '\n'
                # print(f"DEBUG: Sending: {msg.strip()}") 
                self.ser.write(msg.encode("utf-8"))
            except Exception as e:
                print(f"Serial write error: {e}")

    def base_speed_ctrl(self, input_left, input_right):
        # T:1 is the command for speed control
        # Inputs are likely floats (m/s) based on user feedback
        data = {"T":1,"L":input_left,"R":input_right}
        self.send_command(data)

    def gimbal_emergency_stop(self):
        data = {"T":0}
        self.send_command(data)
    
    def close(self):
        if self.ser.is_open:
            self.ser.close()
