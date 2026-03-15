from flask import Flask
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
import os

db = SQLAlchemy()
login_manager = LoginManager()

def create_app():
    template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'templates')
    app = Flask(__name__, template_folder=template_dir)
    app.config['SECRET_KEY'] = 'secret-key-change-in-production'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'routes.login'
    
    from app import models
    from app import routes
    app.register_blueprint(routes.bp)
    
    with app.app_context():
        db.create_all()
        models.create_default_admin()
    
    return app
