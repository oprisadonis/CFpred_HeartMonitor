

from flask import Flask, render_template, redirect, url_for, request, flash, jsonify, Response
from flask_login import login_user, login_required, logout_user, current_user, LoginManager
from flask_socketio import SocketIO

import receiver
from data_models import User, db

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


@app.route('/stopMonitoring')
@login_required
def stopMonitoring():
    print("stopMonitoring")
    receiver.start_stop_signal(start=False)
    return Response(status=204)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
