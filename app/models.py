from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from app import db, login_manager
from datetime import datetime

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    first_name = db.Column(db.String(80), nullable=False)
    last_name = db.Column(db.String(80), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='user')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(200), nullable=False)
    accuracy = db.Column(db.Float, nullable=True)
    loss = db.Column(db.Float, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    results = db.Column(db.Text, nullable=True)
    
    user = db.relationship('User', backref=db.backref('predictions', lazy=True))

class TrainingLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    epoch = db.Column(db.Integer, nullable=False)
    val_accuracy = db.Column(db.Float, nullable=False)
    val_loss = db.Column(db.Float, nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def create_default_admin():
    from app import db
    if not User.query.filter_by(username='admin').first():
        admin = User(
            username='admin',
            first_name='Admin',
            last_name='User',
            role='admin'
        )
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()
