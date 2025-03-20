import json
import socket
import threading

HOST = '192.168.101.13'  # Server address
PORT = 5000  # Port number
CONNECTED = False


# Function to handle client connection
def handle_client():
    print("Waiting for connection...")

    # Send "START" to begin data collection
    conn.sendall("START\n".encode('utf-8'))
    print("Sent START signal")

    buffer = ""

    while True:
        data = conn.recv(1024).decode('utf-8')
        if not data:
            break  # Stop if connection is closed

        buffer += data  # Append received data to buffer

        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)  # Extract one message at a time

            try:
                parsed_data = json.loads(line)
                print(f"Received: {parsed_data}")
            except json.JSONDecodeError:
                print("Error decoding JSON:", line)

    print("Closing connection.")
    conn.close()


# Function to handle the user input for sending the STOP signal
def listen_for_stop_signal():
    while True:
        command = input("Enter 'STOP' to stop the client: ")
        if command.strip().upper() == 'STOP':
            conn.sendall("STOP\n".encode('utf-8'))
            print("Sent STOP signal to client.")
        if command.strip().upper() == 'START':
            conn.sendall("START\n".encode('utf-8'))
            print("Sent START signal to client.")


sock: socket
conn: socket


def main_receiver():
    global CONNECTED, conn, sock
    # Main server code
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((HOST, PORT))
    sock.listen(1)

    conn, addr = sock.accept()
    print(f"Connected by {addr}")
    CONNECTED = True


def start_receiver():
    global CONNECTED
    # Start a separate thread to listen for the STOP command
    stop_thread = threading.Thread(target=listen_for_stop_signal)
    stop_thread.start()

    # Handle client connection in the main thread
    handle_client()

    # Wait for the stop thread to finish before closing the server
    stop_thread.join()

    print("Server is now closed.")
    CONNECTED = False
    sock.close()
