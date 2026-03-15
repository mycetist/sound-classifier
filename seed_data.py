from app import create_app, db
from app.models import User

def seed():
    app = create_app()
    with app.app_context():
        created = []

        if not User.query.filter_by(username='admin').first():
            admin = User(
                username='admin',
                first_name='Admin',
                last_name='User',
                role='admin'
            )
            admin.set_password('admin123')
            db.session.add(admin)
            created.append('  admin / admin123 (admin)')
        else:
            print("Админ уже существует, пропускаем.")

        if not User.query.filter_by(username='user').first():
            user = User(
                username='user',
                first_name='Test',
                last_name='User',
                role='user'
            )
            user.set_password('user123')
            db.session.add(user)
            created.append('  user  / user123  (user)')
        else:
            print("Пользователь уже существует, пропускаем.")

        if created:
            db.session.commit()
            print("Созданны аккаунты:")
            for line in created:
                print(line)

if __name__ == '__main__':
    seed()
