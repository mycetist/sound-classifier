from app import create_app, db
from app.models import User

def seed():
    app = create_app()
    with app.app_context():
        if User.query.filter_by(username='admin').first():
            print("Админ уже существует!")
            return

        admin = User(
            username='admin',
            first_name='Admin',
            last_name='User',
            role='admin'
        )
        admin.set_password('admin123')

        user = User(
            username='user',
            first_name='Test',
            last_name='User',
            role='user'
        )
        user.set_password('user123')

        db.session.add_all([admin, user])
        db.session.commit()
        print("Созданны аккаунты:")
        print("  admin / admin123 (admin)")
        print("  user  / user123  (user)")

if __name__ == '__main__':
    seed()
