import json
import threading
from threading import Lock
from datetime import datetime
import socket
import numpy as np
from queue import Queue
from flask import Flask, render_template, redirect, url_for, request, flash, jsonify, Response, session
from flask_login import login_user, login_required, logout_user, current_user, LoginManager
from flask_socketio import SocketIO, disconnect, join_room, leave_room
from flask_wtf import FlaskForm
from sqlalchemy import func
from wtforms.fields.choices import SelectField
from wtforms.fields.simple import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, EqualTo, Length, Regexp, ValidationError

from CFPrediction import CFPrediction
from fiveMinAnalysis import FiveMinAnalysis
from data_models import User, db, PPGData, SupervisorAccess, PPGFeatures
import neurokit2 as nk

thread = None
thread_lock = Lock()

app = Flask(__name__)

# Set up the database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///heartMonitorDB.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'a534dabbd9f53eda44ebeb65a4743699269d369a9020764280c0d4eb1761beb8'
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

# Initialize the database and login manager
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize CFPrediction
CFP = CFPrediction()


class RegForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    user_type = SelectField('User Type', choices=[('uploader', 'Uploader'), ('supervisor', 'Supervisor')],
                            validators=[DataRequired()], )
    password = PasswordField('Password', validators=[
        DataRequired(),
        Length(min=8, message="Password must be at least 8 characters long."),
        Regexp("^(?=.*?[A-Z])(?=.*?[a-z])(?=.*?[0-9])(?=.*?[^\w\s]).{8,}$",
               message="The password must include one uppercase letter, one lowercase letter, one digit and one special character.")
    ])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('Unavailable username.')


class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')


# # Create the database tables
with app.app_context():
    # db.drop_all()
    db.create_all()

started = False
############################################
HOST = '192.168.101.13'  # Server address
PORT = 5050  # Port number
CONNECTED = False
conn: socket
q = Queue()  # for the live plot in uploader_dashboard
CURRENT_UPLOADER = -1
useFilter = False


# def background_thread():
#     global q
#     filterQ = Queue()
#     while True:
#         if not q.empty():
#             if useFilter:
#                 if filterQ.qsize() >= 4:
#                     filterQ.get()
#                     e = q.get()
#                     filterQ.put(e.red_signal)
#                     avg = sum(list(filterQ.queue)) / filterQ.qsize()
#                     socketio.emit('updateSensorData', {'value': avg, "date": e.timestamp.timestamp()})
#                     filterQ.
#                 else:
#                     e = q.get()
#                     filterQ.put(e.red_signal)
#             else:
#                 e = q.get()
#                 socketio.emit('updateSensorData', {'value': e.red_signal, "date": e.timestamp.timestamp()})
def background_thread():
    # logic of 25Hz for both with or without filter
    global q
    filterQ = []
    tq = []
    dq = []
    while True:
        if not q.empty():
            if useFilter:
                e = q.get()
                filterQ.append(e.red_signal)
                if len(filterQ) >= 4:
                    avg = sum(filterQ) / len(filterQ)
                    socketio.emit('updateSensorData', {'value': [avg], "date": [e.timestamp.timestamp()]})
                    filterQ = []
            else:
                if q.qsize() > 10:
                    for _ in range(10):
                        e = q.get()
                        tq.append(e.timestamp.timestamp())
                        dq.append(e.red_signal)
                    socketio.emit('updateSensorData', {'value': dq, "date": tq})
                    tq = []
                    dq = []


@socketio.on('connect')
def connect():
    global thread
    print('Client connected')
    with thread_lock:
        if thread is None:
            print('THREAD IS NON')
            thread = socketio.start_background_task(background_thread)


@socketio.on('disconnect')
def disconnect():
    print('Client disconnected', request.sid)


skip_counter = 0


# Function to store data in bulk
def save_batch(data_batch, user_id, fma):
    global skip_counter
    skip_counter = skip_counter + 1
    # inserts a batch of PPG records into the database
    global q
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
    for r in records:
        q.put(r)
    if skip_counter > 10:
        fma.addData(records=records)

    print(f"Q: {q.qsize()}")


# Function to handle client connection
def handle_client(user_id):
    """Receives data from the client, processes it, and stores it efficiently."""
    global CONNECTED
    # five minutes analysis Object
    fma = FiveMinAnalysis(db=db, user_id=user_id)
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
                # print(line)
                try:
                    batch_data = json.loads(line)  # Expecting a list of 100 records
                    if isinstance(batch_data, list):
                        save_batch(batch_data, user_id, fma)  # Store all 100 records at once
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
    global CONNECTED, conn, CURRENT_UPLOADER
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((HOST, PORT))
    sock.listen(1)
    conn, addr = sock.accept()
    print(f"Connected by {addr}")
    CONNECTED = True
    CURRENT_UPLOADER = session['userid']


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
    if CONNECTED:
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
    f = RegForm()
    if f.validate_on_submit():
        username = f.username.data
        user_type = f.user_type.data
        password = f.password.data

        # create new user
        new_user = User(username=username, user_type=user_type)
        new_user.set_password(password)
        # insert new_user into the database
        db.session.add(new_user)
        db.session.commit()

        flash("User successfully registered!")
        return redirect(url_for('login'))

    return render_template('signup.html', f=f)


@app.route('/login', methods=['GET', 'POST'])
def login():
    f = LoginForm()
    if f.validate_on_submit():
        username = f.username.data
        password = f.password.data

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)  # Log the user in
            session['userid'] = current_user.id
            return redirect(url_for('dashboard'))  # Redirect to the dashboard after successful login
        else:
            flash('Invalid username or password')

    return render_template('login.html', f=f)


@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.role == 'uploader':
        supervisor_accesses = SupervisorAccess.query.filter_by(uploader_id=current_user.id).all()
        drawer_open = request.args.get('drawer_open', False)
        return render_template('uploader_dashboard.html', supervisor_accesses=supervisor_accesses,
                               drawer_open=drawer_open)  # Redirect to uploader dashboard
    elif current_user.role == 'supervisor':
        uploaders = SupervisorAccess.query.filter_by(supervisor_id=current_user.id).all()
        return render_template('supervisor_dashboard.html', uploaders=uploaders)  # Redirect to supervisor dashboard
    else:
        flash("Unauthorized access!")
        return redirect(url_for('home'))


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))


@app.route('/connectSensor')
@login_required
def connectSensor():
    main_receiver()


@app.route("/is_user_active/<userid>")
def is_user_active(userid):
    print(f'userid: {userid}\ncurrent_uploader: {CURRENT_UPLOADER}')
    print(f'userid: {userid} (type: {type(userid)})')
    print(f'CURRENT_UPLOADER: {CURRENT_UPLOADER} (type: {type(CURRENT_UPLOADER)})')
    if int(userid) == int(CURRENT_UPLOADER):
        is_active = True
    else:
        is_active = False
    return jsonify({"active": is_active})


@app.route('/startMonitoring')
@login_required
def startMonitoring():
    global started
    print("startMonitoring")
    if not started:
        start_receiver(current_user.id)
        started = True
    else:
        start_stop_signal(start=True)
    return Response(status=204)


@app.route('/stopMonitoring')
@login_required
def stopMonitoring():
    print("stopMonitoring")
    start_stop_signal(start=False)
    return Response(status=204)


##################################################################
@app.route('/add_supervisor_access', methods=['POST'])
@login_required
def add_supervisor_access():
    username = request.form['username']
    supervisor = User.query.filter_by(username=username, role='supervisor').first()

    if not supervisor:
        flash('Supervisor not found in the system.', 'error')
        return redirect(url_for('dashboard', drawer_open=True))

    # search for the access
    existing_access = SupervisorAccess.query.filter_by(
        uploader_id=current_user.id,
        supervisor_id=supervisor.id
    ).first()

    if existing_access:
        flash('Access already granted.', 'warning')
        return redirect(url_for('dashboard', drawer_open=True))

    # Add new relationship
    access = SupervisorAccess(uploader_id=current_user.id, supervisor_id=supervisor.id)
    db.session.add(access)
    db.session.commit()
    flash('Supervisor access granted.', 'success')
    return redirect(url_for('dashboard', drawer_open=True))


@app.route('/remove_supervisor_access/<int:supervisor_id>', methods=['POST'])
@login_required
def remove_supervisor_access(supervisor_id):
    access = SupervisorAccess.query.filter_by(
        uploader_id=current_user.id,
        supervisor_id=supervisor_id
    ).first()

    if not access:
        flash('Supervisor access not found.', 'error')
        return redirect(url_for('dashboard', drawer_open=True))

    db.session.delete(access)
    db.session.commit()
    flash('Supervisor access removed.', 'success')
    return redirect(url_for('dashboard', drawer_open=True))


@app.route('/activateFilter')
@login_required
def activateFilter():
    global useFilter
    useFilter = not useFilter


@app.route("/get_dates/<int:user_id>", methods=["GET"])
@login_required
def get_dates(user_id):
    unique_dates = (
        db.session.query(func.date(PPGFeatures.start_time))
        .filter(PPGFeatures.user_id == int(user_id))
        .distinct()
        .order_by(func.date(PPGFeatures.start_time))
        .all()
    )
    dates = [d[0] for d in unique_dates]
    return jsonify({"dates": dates})


# def show_ppg_features():
#     records = PPGFeatures.query.all()
#     signal = PPGData.query.filter(PPGData.timestamp >= records[2].start_time, PPGData.timestamp <= records[2].finish_time).all()
#     red_signal = [r.red_signal for r in signal]

@app.route("/get_feature_for_date/<feature>/<date>/<selectedUploaderId>")
def get_feature_for_date(feature, date, selectedUploaderId):
    date_formatted = datetime.strptime(date, "%Y-%m-%d").date()
    column = getattr(PPGFeatures, feature)
    features = (PPGFeatures
                .query
                .with_entities(PPGFeatures.finish_time, column)
                .filter(PPGFeatures.user_id == int(selectedUploaderId))
                .filter(func.date(PPGFeatures.finish_time) == date_formatted)
                .all())
    formatted_feature = [(dt.strftime("%H:%M"), float("{:.2f}".format(f))) for dt, f in features]
    return jsonify({"data": formatted_feature})


@app.route("/cognitive_fatigue_prediction/<date>/<uploaderId>")
def cognitive_fatigue_prediction(date, uploaderId):
    date_formatted = datetime.strptime(date, "%Y-%m-%d").date()
    predictions = CFP.predict(db.engine, int(uploaderId), date_formatted)
    # finish_time = ["12:23", "12:45", "12:56", "12:59"]
    # fatigue = [0,1,0,1]
    # import pandas as pd
    # df = pd.DataFrame({
    #     'finish_time': finish_time,
    #     'fatigue': fatigue
    # })

    return jsonify({"data": predictions.to_dict(orient="records")})


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
