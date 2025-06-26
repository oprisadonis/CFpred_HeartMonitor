import socket
import json
import time
import max30102
import threading
import atexit
import sys

m = max30102.MAX30102()

# socket connection
HOST = '' # put host address
#HOST = ''
PORT = 5050
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))
print("Waiting for START signal...")

b = ""  # buffer
send = False

# data collection function
def collect_data():
    global send
    batch_size = 100  
    batch_data = []  # list to accumulate records

    while True:
        if send:
            #st = time.time()
            red, ir = m.read_sequential(100)
            start_time = time.time()
            for i in range(len(red)):
                t = start_time + i* 0.01
                data = {
                    "t": t,
                    "red": red[i],
                    "ir": ir[i],
                }
                batch_data.append(data)

            if len(batch_data) == batch_size:
                json_data = json.dumps(batch_data) + "\n"
                s.sendall(json_data.encode('utf-8'))
                batch_data = []  # Reset batch
                #print(st-time.time())


def socket_communication():
    global send, b
    while True:
        data = s.recv(4096).decode('utf-8')
        if not data:
            break  # Stop if connection closed
        b += data
        while "\n" in b:
            line, b = b.split("\n", 1)
            if line == "START":
                send = True
                print("START sending data")
                # Start sensor
                m.reset()
                m.setup()  
                time.sleep(1)
            elif line == "STOP":
                send = False
                print("STOP data uploading stopped")
                # Stop sensor
                m.shutdown()

# Start the socket communication in a separate thread
thread_socket = threading.Thread(target=socket_communication)
thread_socket.daemon = True
thread_socket.start()

# Start the data collection in a separate thread
thread_data = threading.Thread(target=collect_data)
thread_data.daemon = True
thread_data.start()

def shutdown_sensor():
    if m:
        m.shutdown()  
    print("Sensor shutdown completed")

atexit.register(shutdown_sensor)

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass

s.close()
print("Connection closed")
