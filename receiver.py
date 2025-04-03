import json
import socket
import threading
from datetime import datetime

from flask_socketio import SocketIO
from flask import current_app

from data_models import PPGData, db

HOST = '192.168.101.13'  # Server address
PORT = 5050  # Port number
CONNECTED = False

sock: socket
conn: socket


# Function to store data in bulk
def save_batch(data_batch, user_id):
    """Efficiently inserts a batch of PPG records into the database."""
    records = [
        PPGData(
            user_id=user_id,
            timestamp=datetime.fromtimestamp(data['t']),
            red_signal=data['red'],
            ir_signal=data['ir']
        ) for data in data_batch
    ]

    db.session.bulk_save_objects(records)  # Bulk insert
    db.session.commit()
    print(f"Inserted {len(records)} records into the database.")

    # Emit data to connected clients in Flask context
    from app import app, socketio
    with app.app_context():
        socketio.emit('ppg_data', {
            'timestamps': [r.timestamp.timestamp() for r in records],
            'red_signals': [r.red_signal for r in records],
            'ir_signals': [r.ir_signal for r in records]
        })


# Function to handle client connection
# Function to handle client connection
def handle_client(user_id):
    """Receives data from the client, processes it, and stores it efficiently."""
    global CONNECTED, conn
    print("Waiting for connection...")
    from app import app
    with app.app_context():
        start_stop_signal(start=True)

        buffer = ""

        while True:
            data = conn.recv(4096).decode('utf-8')  # Larger buffer for batch data
            if not data:
                break  # Stop if connection is closed

            buffer += data  # Append received data to buffer

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)  # Extract one message at a time
                print(line)
                try:
                    batch_data = json.loads(line)  # Expecting a list of 100 records
                    if isinstance(batch_data, list):
                        save_batch(batch_data, user_id)  # Store all 100 records at once
                    else:
                        print("Error: Expected a list of records.")
                except json.JSONDecodeError:
                    print("Error decoding JSON:", line)

        print("Closing connection.")
        conn.close()


# Function to start/stop monitoring
def start_stop_signal(start):
    if start:
        conn.sendall("START\n".encode('utf-8'))
        print("Sent START signal to client.")
    else:
        conn.sendall("STOP\n".encode('utf-8'))
        print("Sent STOP signal to client.")


# Main receiver function
def main_receiver():
    global CONNECTED, conn, sock
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((HOST, PORT))
    sock.listen(1)
    conn, addr = sock.accept()
    print(f"Connected by {addr}")
    CONNECTED = True


# Function to start receiver from Flask
def start_receiver(user_id):
    global CONNECTED
    threading.Thread(target=handle_client, args=(user_id,), daemon=True).start()
    print("Receiver started in a new thread.")
