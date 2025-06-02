from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

db = SQLAlchemy()


# Define the User model for the database
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(150), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # "uploader" or "supervisor"

    def __init__(self, username, user_type):
        self.username = username
        self.role = user_type

    def __repr__(self):
        return f'<User {self.username} - {self.role}>'

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)


class PPGData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)  # Foreign key for user
    timestamp = db.Column(db.DateTime, nullable=False)
    red_signal = db.Column(db.Float, nullable=False)
    ir_signal = db.Column(db.Float, nullable=False)

    # Relationship to User model
    user = db.relationship('User', backref=db.backref('ppg_data', lazy=True))

    def __repr__(self):
        return f"<PPGData {self.id} - User {self.user.username} at {self.timestamp}>"


class PPGFeatures(db.Model):
    __tablename__ = 'ppg_features'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    start_time = db.Column(db.DateTime, nullable=False)
    finish_time = db.Column(db.DateTime, nullable=False)
    fatigue = db.Column(db.Boolean, nullable=True)

    bpm = db.Column(db.Float, nullable=False)
    ibi = db.Column(db.Float, nullable=False)
    sdnn = db.Column(db.Float, nullable=False)
    sdsd = db.Column(db.Float, nullable=False)
    rmssd = db.Column(db.Float, nullable=False)
    pnn20 = db.Column(db.Float, nullable=False)
    pnn50 = db.Column(db.Float, nullable=False)
    hr_mad = db.Column(db.Float, nullable=False)
    sd1 = db.Column(db.Float, nullable=False)
    sd2 = db.Column(db.Float, nullable=False)
    s = db.Column(db.Float, nullable=False)
    sd1_sd2 = db.Column(db.Float, nullable=False)  # sd1/sd2
    breathingrate = db.Column(db.Float, nullable=False)
    vlf = db.Column(db.Float, nullable=False)
    lf = db.Column(db.Float, nullable=False)
    hf = db.Column(db.Float, nullable=False)
    lf_hf = db.Column(db.Float, nullable=False)  # lf/hf
    p_total = db.Column(db.Float, nullable=False)
    vlf_perc = db.Column(db.Float, nullable=False)
    lf_perc = db.Column(db.Float, nullable=False)
    hf_perc = db.Column(db.Float, nullable=False)

    # Relationship to User model
    user = db.relationship('User', backref=db.backref('ppg_features', lazy=True))


class SupervisorAccess(db.Model):
    __tablename__ = 'supervisor_access'

    uploader_id = db.Column(db.Integer, db.ForeignKey('user.id'), primary_key=True)
    supervisor_id = db.Column(db.Integer, db.ForeignKey('user.id'), primary_key=True)

    # Relationships to User model
    uploader = db.relationship('User', foreign_keys=[uploader_id], backref='granted_supervisors')
    supervisor = db.relationship('User', foreign_keys=[supervisor_id], backref='accessible_uploaders')

    def __repr__(self):
        return f"<Access uploader={self.uploader_id} supervisor={self.supervisor_id}>"
