import json
import threading
from datetime import datetime
import socket

from flask import Flask, render_template, redirect, url_for, request, flash, jsonify, Response
from flask_login import login_user, login_required, logout_user, current_user, LoginManager
from flask_socketio import SocketIO, send, disconnect, join_room, leave_room

import receiver
from data_models import User, db, PPGData

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Set up the database (using SQLite in this example)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///heartMonitorDB.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'a534dabbd9f53eda44ebeb65a4743699269d369a9020764280c0d4eb1761beb8'

# Initialize the database and login manager
db.init_app(app)  # Initialize db with app
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Create the database tables
# with app.app_context():
# db.drop_all()
# db.create_all()

started = False
############################################
HOST = '192.168.101.13'  # Server address
PORT = 5050  # Port number
CONNECTED = False
conn: socket


def send_to_client(records):
    if not records:
        return

    user_id = records[0].user_id
    room = f"user_{user_id}"

    payload = {
        'timestamps': [r.timestamp.timestamp() for r in records],
        'red_signals': [r.red_signal for r in records],
        'ir_signals': [r.ir_signal for r in records]
    }

    socketio.emit('ppg_data', payload, room=room)
    print(f"Sent {len(records)} PPG records to user {user_id}")


@socketio.on('connect')
def handle_connect():
    if not current_user.is_authenticated:
        print("Unauthorized WebSocket connection.")
        return disconnect()

    user_id = current_user.id
    room = f"user_{user_id}"  # Create a unique room
    # Add user to their room
    join_room(room)

    print(f"User {user_id} connected and joined room {room}")


@socketio.on('disconnect')
def handle_disconnect():
    if current_user.is_authenticated:
        user_id = current_user.id
        room = f"user_{user_id}"
        leave_room(room)

    print(f"User {current_user.id} disconnected.")


# Function to store data in bulk
def save_batch(data_batch, user_id):
    # inserts a batch of PPG records into the database
    records = [
        PPGData(
            user_id=user_id,
            timestamp=datetime.fromtimestamp(data['t']),
            red_signal=data['red'],
            ir_signal=data['ir']
        ) for data in data_batch
    ]

    db.session.bulk_save_objects(records)
    db.session.commit()
    print(f"Inserted {len(records)} records into the database.")

    send_to_client(records)


# Function to handle client connection
def handle_client(user_id):
    """Receives data from the client, processes it, and stores it efficiently."""
    global CONNECTED
    print("Waiting for connection...")
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
    global CONNECTED, conn
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


############################################

# Load user for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.cli.command("reset_db")
def reset_db():
    """Reset the database."""
    db.drop_all()
    db.create_all()
    print("Database has been reset!")


# Connection & sensor check
def check_sensor_status():
    if receiver.CONNECTED:
        return "connected"
    else:
        return "disconnected"


@app.route('/sensor_status')
def sensor_status():
    status = check_sensor_status()
    return jsonify({"status": status})


# Step 2: Routes for Signup, Login, and Dashboard
@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))  # Automatically go to the dashboard if logged in
    return render_template('index.html')  # Otherwise, show the home page with login/sign-up options


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        user_type = request.form['user_type']  # Get user role from form

        if password != confirm_password:
            flash("Passwords do not match!")
            return redirect(url_for('signup'))

        # Check if username exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash("Username already exists!")
            return redirect(url_for('signup'))

        # Create new user
        new_user = User(username=username, role=user_type)
        new_user.set_password(password)

        db.session.add(new_user)
        db.session.commit()

        flash("User successfully registered!")
        return redirect(url_for('login'))

    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)  # Log the user in
            return redirect(url_for('dashboard'))  # Redirect to the dashboard after successful login
        else:
            flash('Invalid username or password')

    return render_template('login.html')


@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.role == 'uploader':
        receiver.main_receiver()
        return render_template('uploader_dashboard.html')  # Redirect to uploader dashboard
    elif current_user.role == 'supervisor':
        return render_template('supervisor_dashboard.html')  # Redirect to supervisor dashboard
    else:
        flash("Unauthorized access!")
        return redirect(url_for('home'))


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))


@app.route('/startMonitoring')
@login_required
def startMonitoring():
    global started
    print("startMonitoring")
    if not started:
        receiver.start_receiver(current_user.id)
        started = True
    else:
        receiver.start_stop_signal(start=True)
    return Response(status=204)


@socketio.on('connect')
def handle_connect():
    print("Client connected")


@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")


@app.route('/stopMonitoring')
@login_required
def stopMonitoring():
    print("stopMonitoring")
    receiver.start_stop_signal(start=False)
    return Response(status=204)


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
