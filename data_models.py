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
